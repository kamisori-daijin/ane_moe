//
//  RouterContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation

/// A hardware-accelerated routing manager that dispatches token sequences to specific experts based on gating probabilities.
public final class RouterContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let expertCount = 256
    public let topK = 2
    public let hiddenDimensions = 2048
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private var layers: [Int: MLModel] = [:]
    
    // Global shared execution workspaces to strictly control the memory footprint across layers
    private let sharedExpertInputs: [Int: CVPixelBuffer]
    private let sharedExpertOutputs: [Int: CVPixelBuffer]
    
    // MARK: - Initialization
    
    /// Initializes the router by scanning the specified directory and allocating shared hardware registers.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 24) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        let fileManager = FileManager.default
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        
        let totalBufferSize = maxSequenceLength * hiddenDimensions
        
        var inputs: [Int: CVPixelBuffer] = [:]
        var outputs: [Int: CVPixelBuffer] = [:]
        
        for expertIndex in 0..<expertCount {
            var inputBuffer: CVPixelBuffer?
            var outputBuffer: CVPixelBuffer?
            
            CVPixelBufferCreate(kCFAllocatorDefault, totalBufferSize, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &inputBuffer)
            CVPixelBufferCreate(kCFAllocatorDefault, totalBufferSize, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &outputBuffer)
            
            guard let resolvedInput = inputBuffer, let resolvedOutput = outputBuffer else {
                throw NSError(domain: "MoERouter", code: -1, userInfo: [NSLocalizedDescriptionKey: "Hardware allocation failed for shared expert pools."])
            }
            inputs[expertIndex] = resolvedInput
            outputs[expertIndex] = resolvedOutput
        }
        
        self.sharedExpertInputs = inputs
        self.sharedExpertOutputs = outputs
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
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
    /// - Returns: The mutated sequence register containing the scaled expert combination outputs.
    @discardableResult
    public func route(_ hiddenStates: CVPixelBuffer, layerIndex: Int, activeTokenCount tokenCount: Int) async throws -> CVPixelBuffer {
        guard let routerModel = layers[layerIndex] else { return hiddenStates }
        
        // 1. Evaluate Gating Probabilities
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(pixelBuffer: hiddenStates)
        ])
        let prediction = try await routerModel.prediction(from: features)
        
        guard let scoresArray = prediction.featureValue(for: "router_scores")?.multiArrayValue,
              let indicesArray = prediction.featureValue(for: "router_indices")?.multiArrayValue else {
            throw NSError(domain: "MoERouter", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid gating outputs."])
        }
        
        // 2. Track Routing Metrics
        var expertTokenCounts = [Int](repeating: 0, count: expertCount)
        var routingMap = [[(expertIndex: Int, position: Int)]](repeating: [], count: tokenCount)
        
     
        let indicesStrides = indicesArray.strides.map { $0.intValue }
        let indicesRawPtr = indicesArray.dataPointer.assumingMemoryBound(to: Int32.self)
        
        CVPixelBufferLockBaseAddress(hiddenStates, .readOnly)
        let sourcePointer = CVPixelBufferGetBaseAddress(hiddenStates)!.assumingMemoryBound(to: Float16.self)
        
        // 3. Scatter Step: Distribute tokens into expert slots
        for tokenIndex in 0..<tokenCount {
            for k in 0..<topK {
                
                let offset = (tokenIndex * indicesStrides[1]) + (k * indicesStrides[2])
                let expertIndex = Int(indicesRawPtr[offset])
                
                let currentPosition = expertTokenCounts[expertIndex]
                routingMap[tokenIndex].append((expertIndex, currentPosition))
                
                if let expertInput = sharedExpertInputs[expertIndex] {
                    CVPixelBufferLockBaseAddress(expertInput, [])
                    let destinationPointer = CVPixelBufferGetBaseAddress(expertInput)!.assumingMemoryBound(to: Float16.self)
                    
                    let sourceOffset = tokenIndex * hiddenDimensions
                    let destinationOffset = currentPosition * hiddenDimensions
                    
                    memcpy(destinationPointer.advanced(by: destinationOffset),
                           sourcePointer.advanced(by: sourceOffset),
                           hiddenDimensions * MemoryLayout<Float16>.stride)
                    
                    CVPixelBufferUnlockBaseAddress(expertInput, [])
                }
                expertTokenCounts[expertIndex] += 1
            }
        }
        CVPixelBufferUnlockBaseAddress(hiddenStates, .readOnly)
        
      
        // 5. Gather Step: Merge, scale, and accumulate localized results
        let scoresStrides = scoresArray.strides.map { $0.intValue }
        let scoresRawPtr = scoresArray.dataPointer.assumingMemoryBound(to: Float.self)
        
        CVPixelBufferLockBaseAddress(hiddenStates, [])
        let finalDestinationPointer = CVPixelBufferGetBaseAddress(hiddenStates)!.assumingMemoryBound(to: Float16.self)
        memset(finalDestinationPointer, 0, tokenCount * hiddenDimensions * MemoryLayout<Float16>.stride)
        
        for tokenIndex in 0..<tokenCount {
            for k in 0..<topK {
                let routing = routingMap[tokenIndex][k]
                
                let offset = (tokenIndex * scoresStrides[1]) + (k * scoresStrides[2])
                let score = Float16(scoresRawPtr[offset])
                
                if let expertOutput = sharedExpertOutputs[routing.expertIndex] {
                    CVPixelBufferLockBaseAddress(expertOutput, .readOnly)
                    let expertSourcePointer = CVPixelBufferGetBaseAddress(expertOutput)!.assumingMemoryBound(to: Float16.self)
                    
                    let sourceOffset = routing.position * hiddenDimensions
                    let destinationOffset = tokenIndex * hiddenDimensions
                    
                    for d in 0..<hiddenDimensions {
                        finalDestinationPointer[destinationOffset + d] += expertSourcePointer[sourceOffset + d] * score
                    }
                    CVPixelBufferUnlockBaseAddress(expertOutput, .readOnly)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(hiddenStates, [])
        
        return hiddenStates
    }
}
