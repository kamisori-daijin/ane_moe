//
//  RouterContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//
import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated routing manager that dispatches token sequences to specific experts based on gating probabilities.
public final class RouterContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let expertCount = 256
    public let topK = 2
    public let hiddenDimensions = 2048
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private var layers: [Int: MLModel] = [:]
    
    // Shared expert workspaces managed as IOSurface-backed MLMultiArrays
    public let sharedExpertInputs: [Int: MLMultiArray]
    public let sharedExpertOutputs: [Int: MLMultiArray]
    
    // MARK: - Initialization
    
    /// Initializes the router by scanning the specified directory and allocating shared hardware registers.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        let fileManager = FileManager.default
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        // Exact shape definition fully aligned with Python's converter layout: [1, Tokens(512), HiddenDim(2048)]
        let expertShape: [NSNumber] = [1, maxSequenceLength as NSNumber, hiddenDimensions as NSNumber]
        
        var inputs: [Int: MLMultiArray] = [:]
        var outputs: [Int: MLMultiArray] = [:]
        
        for expertIndex in 0..<expertCount {
            var inputBuffer: CVPixelBuffer? = nil
            var outputBuffer: CVPixelBuffer? = nil
            
            // Map pixel buffer tracking dimensions straight to tensor shape layouts.
            // Width maps to the trailing dimension (2048), Height maps to the sequence slots (512).
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                hiddenDimensions,  // Width = 2048
                maxSequenceLength, // Height = 512
                kCVPixelFormatType_OneComponent16Half,
                bufferAttributes as CFDictionary,
                &inputBuffer
            )
            
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                hiddenDimensions,
                maxSequenceLength,
                kCVPixelFormatType_OneComponent16Half,
                bufferAttributes as CFDictionary,
                &outputBuffer
            )
            
            guard let resolvedInput = inputBuffer, let resolvedOutput = outputBuffer else {
                throw NSError(
                    domain: "MoERouter",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Hardware allocation failed for shared expert pools."]
                )
            }
            
            // Zero-copy multi-array bindings that now fully clear OS hardware-native stride validations.
            inputs[expertIndex] = MLMultiArray(pixelBuffer: resolvedInput, shape: expertShape)
            outputs[expertIndex] = MLMultiArray(pixelBuffer: resolvedOutput, shape: expertShape)
        }
        
        self.sharedExpertInputs = inputs
        self.sharedExpertOutputs = outputs
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else {
                continue
            }
            
            let routerModelURL = layerDirectoryURL.appendingPathComponent("router.mlmodelc")
            if fileManager.fileExists(atPath: routerModelURL.path) {
                self.layers[layerIndex] = try await MLModel.load(contentsOf: routerModelURL, configuration: configuration)
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Routes the incoming hidden states through the gating pipeline, scattering and gathering tokens via hardware registers.
    ///
    /// - Parameters:
    ///   - hiddenStates: The input tracking sequence register containing current spatial states.
    ///   - layerIndex: The current executing structural block index inside the network graph.
    ///   - tokenCount: The factual sequence length currently active inside the context window.
    ///   - options: The dynamic prediction options tracking memory output backings.
    /// - Returns: The mutated sequence register containing the scaled expert combination outputs.
    @discardableResult
    public func route(
        _ hiddenStates: MLMultiArray,
        layerIndex: Int,
        activeTokenCount tokenCount: Int,
        options: MLPredictionOptions
    ) async throws -> (outputTensor: MLMultiArray, activeExperts: Set<Int>) {
        guard let routerModel = layers[layerIndex] else {
            return (hiddenStates, [])
        }
        
        // 1. Evaluate Gating Probabilities
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStates)
        ])
        let prediction = try await routerModel.prediction(from: features, options: options)
        
        guard let scoresArray = prediction.featureValue(for: "router_scores")?.multiArrayValue,
              let indicesArray = prediction.featureValue(for: "router_indices")?.multiArrayValue else {
            throw NSError(
                domain: "MoERouter",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Invalid gating outputs."]
            )
        }
        
        // 2. Track Routing Metrics
        var expertTokenCounts = [Int](repeating: 0, count: expertCount)
        var routingMap = [[(expertIndex: Int, position: Int)]](repeating: [], count: tokenCount)
        var activeExperts = Set<Int>()
        
        let indicesStrides = indicesArray.strides.map { $0.intValue }
        let indicesRawPtr = indicesArray.dataPointer.assumingMemoryBound(to: Int32.self)
        
        // Lock the source pixel buffer for safe pointer access
        guard let srcPixelBuffer = hiddenStates.pixelBuffer else {
            return (hiddenStates, [])
        }
        CVPixelBufferLockBaseAddress(srcPixelBuffer, .readOnly)
        let sourcePointer = CVPixelBufferGetBaseAddress(srcPixelBuffer)!.assumingMemoryBound(to: Float16.self)
        
        // 3. Scatter Step: Distribute tokens into expert slots
        for tokenIndex in 0..<tokenCount {
            for k in 0..<topK {
                let offset = (tokenIndex * indicesStrides[1]) + (k * indicesStrides[2])
                let expertIndex = Int(indicesRawPtr[offset])
                
                let currentPosition = expertTokenCounts[expertIndex]
                routingMap[tokenIndex].append((expertIndex, currentPosition))
                activeExperts.insert(expertIndex) // Mark active expert IDs for dynamic loading
                
                if let expertInputTensor = sharedExpertInputs[expertIndex],
                   let expertInputBuffer = expertInputTensor.pixelBuffer {
                    CVPixelBufferLockBaseAddress(expertInputBuffer, [])
                    let destinationPointer = CVPixelBufferGetBaseAddress(expertInputBuffer)!.assumingMemoryBound(to: Float16.self)
                    
                    let sourceOffset = tokenIndex * hiddenDimensions
                    let destinationOffset = currentPosition * hiddenDimensions
                    
                    memcpy(
                        destinationPointer.advanced(by: destinationOffset),
                        sourcePointer.advanced(by: sourceOffset),
                        hiddenDimensions * MemoryLayout<Float16>.stride
                    )
                    
                    CVPixelBufferUnlockBaseAddress(expertInputBuffer, [])
                }
                expertTokenCounts[expertIndex] += 1
            }
        }
        CVPixelBufferUnlockBaseAddress(srcPixelBuffer, .readOnly)
        
        // 4. Control returns to pipeline to invoke ExpertsContainer.executeRequiredChunks using activeExperts
        
        // 5. Gather Step: Merge, scale, and accumulate localized results
        let scoresStrides = scoresArray.strides.map { $0.intValue }
        let scoresRawPtr = scoresArray.dataPointer.assumingMemoryBound(to: Float.self)
        
        CVPixelBufferLockBaseAddress(srcPixelBuffer, [])
        let finalDestinationPointer = CVPixelBufferGetBaseAddress(srcPixelBuffer)!.assumingMemoryBound(to: Float16.self)
        memset(finalDestinationPointer, 0, tokenCount * hiddenDimensions * MemoryLayout<Float16>.stride)
        
        for tokenIndex in 0..<tokenCount {
            for k in 0..<topK {
                let routeInfo = routingMap[tokenIndex][k]
                let expertIndex = routeInfo.expertIndex
                let currentPosition = routeInfo.position
                
                let scoreOffset = (tokenIndex * scoresStrides[1]) + (k * scoresStrides[2])
                let gateScore = scoresRawPtr[scoreOffset]
                
                if let expertOutputTensor = sharedExpertOutputs[expertIndex],
                   let expertOutputBuffer = expertOutputTensor.pixelBuffer {
                    CVPixelBufferLockBaseAddress(expertOutputBuffer, .readOnly)
                    let sourcePointer = CVPixelBufferGetBaseAddress(expertOutputBuffer)!.assumingMemoryBound(to: Float16.self)
                    
                    let srcOffset = currentPosition * hiddenDimensions
                    let dstOffset = tokenIndex * hiddenDimensions
                    
                    let srcTokenPtr = sourcePointer.advanced(by: srcOffset)
                    let dstTokenPtr = finalDestinationPointer.advanced(by: dstOffset)
                    
                    for d in 0..<hiddenDimensions {
                        dstTokenPtr[d] += Float16(Float(srcTokenPtr[d]) * gateScore)
                    }
                    CVPixelBufferUnlockBaseAddress(expertOutputBuffer, .readOnly)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(srcPixelBuffer, [])
        return (hiddenStates, activeExperts)
    }
}
