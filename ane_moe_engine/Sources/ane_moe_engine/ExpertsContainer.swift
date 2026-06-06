//
//  ExpertsContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation

/// A hardware-accelerated execution container that dynamically loads, executes, and unloads 4-expert quantized slices.
public final class ExpertsContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let expertCount = 256
    public let chunkSize = 4
    public let hiddenDimensions = 2048
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private let baseDirectoryURL: URL
    private let modelConfiguration: MLModelConfiguration
    
    // MARK: - Initialization
    
    /// Initializes the expert execution pipeline linked to targeted physical model paths.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 24) {
        self.totalLayers = totalLayers
        self.baseDirectoryURL = baseDirectoryURL
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        self.modelConfiguration = configuration
    }
    
    // MARK: - Public Execution API
    
    /// Dynamically discovers, serializes, and evaluates localized expert chunks based on factual active routing sets.
    ///
    /// - Parameters:
    ///   - layerIndex: The current executing structural block index inside the network graph.
    ///   - activeExperts: A distinct set of globally assigned expert IDs flagged for computation.
    ///   - sharedInputs: The global shared memory mapping registry containing routed inputs.
    ///   - sharedOutputs: The target registry designated to capture computed output lanes.
    public func executeRequiredChunks(
        layerIndex: Int,
        activeExperts: Set<Int>,
        sharedInputs: [Int: CVPixelBuffer],
        sharedOutputs: [Int: CVPixelBuffer]
    ) async throws {
        let fileManager = FileManager.default
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        
        // Match standard 3D tracking layouts: [chunkSize(4), tokens(1), 2048]
        let chunkBufferSize = chunkSize * 1 * hiddenDimensions
        let singleExpertSliceBytes = 1 * hiddenDimensions * MemoryLayout<Float16>.stride
        
        // Isolate and unify target tracking starts inside unique set maps
        let requiredChunkStarts = Set(activeExperts.map { ($0 / chunkSize) * chunkSize })
        
        for startIndex in requiredChunkStarts {
            let chunkName = String(format: "qwen_expert_layer_%d_chunk_%03d.mlmodelc", layerIndex, startIndex)
            let chunkURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)").appendingPathComponent(chunkName)
            
            guard fileManager.fileExists(atPath: chunkURL.path) else { continue }
            
            // --- 1. Dynamic Model Load ---
            // On-demand loading of the serialized CoreML model graph into memory
            let chunkModel = try await MLModel.load(contentsOf: chunkURL, configuration: modelConfiguration)
            
            // Allocate transient hardware workspaces local to the iteration block
            var inputWorkspace: CVPixelBuffer?
            var outputWorkspace: CVPixelBuffer?
            
            CVPixelBufferCreate(kCFAllocatorDefault, chunkBufferSize, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &inputWorkspace)
            CVPixelBufferCreate(kCFAllocatorDefault, chunkBufferSize, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &outputWorkspace)
            
            guard let resolvedInput = inputWorkspace, let resolvedOutput = outputWorkspace else { continue }
            
            // --- 2. Scatter Packing Phase ---
            // Pack fragmented tokens from the 4 active experts into a contiguous input buffer
            CVPixelBufferLockBaseAddress(resolvedInput, [])
            let destinationPointer = CVPixelBufferGetBaseAddress(resolvedInput)!.assumingMemoryBound(to: Float16.self)
            
            for offset in 0..<chunkSize {
                let globalExpertIndex = startIndex + offset
                
                if let individualInput = sharedInputs[globalExpertIndex] {
                    CVPixelBufferLockBaseAddress(individualInput, .readOnly)
                    let sourcePointer = CVPixelBufferGetBaseAddress(individualInput)!.assumingMemoryBound(to: Float16.self)
                    
                    let targetOffset = offset * 1 * hiddenDimensions
                    memcpy(destinationPointer.advanced(by: targetOffset), sourcePointer, singleExpertSliceBytes)
                    
                    CVPixelBufferUnlockBaseAddress(individualInput, .readOnly)
                }
            }
            CVPixelBufferUnlockBaseAddress(resolvedInput, [])
            
            // --- 3. Hardware Computation Execution ---
            // Execute safe asynchronous batch inference on CoreML without illegal state provider arguments
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "expert_batched_hidden_states": MLFeatureValue(pixelBuffer: resolvedInput)
            ])
            
            let inferenceResult = try await chunkModel.prediction(from: inputFeatures)
            
            guard let computedOutputFeature = inferenceResult.featureValue(for: "down_out"),
                  let computedPixelBuffer = computedOutputFeature.imageBufferValue else {
                throw NSError(domain: "MoEExpertPipeline", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid expert feature outputs returned from CoreML graph."])
            }
            
            // --- 4. Gather Unpacking Phase ---
            // Unpack and distribute unified batch outputs back into the 256 individual registers
            CVPixelBufferLockBaseAddress(computedPixelBuffer, .readOnly)
            let globalSourcePointer = CVPixelBufferGetBaseAddress(computedPixelBuffer)!.assumingMemoryBound(to: Float16.self)
            
            for offset in 0..<chunkSize {
                let globalExpertIndex = startIndex + offset
                
                if let individualOutput = sharedOutputs[globalExpertIndex] {
                    CVPixelBufferLockBaseAddress(individualOutput, [])
                    let individualDestinationPointer = CVPixelBufferGetBaseAddress(individualOutput)!.assumingMemoryBound(to: Float16.self)
                    
                    let sourceOffset = offset * 1 * hiddenDimensions
                    memcpy(individualDestinationPointer, globalSourcePointer.advanced(by: sourceOffset), singleExpertSliceBytes)
                    
                    CVPixelBufferUnlockBaseAddress(individualOutput, [])
                }
            }
            CVPixelBufferUnlockBaseAddress(computedPixelBuffer, .readOnly)
            
            // --- 5. Automatic Memory Purge ---
            // Execution context out of bounds. `chunkModel`, transient arrays, and hardware maps are instantly evicted by Swift ARC.
        }
    }
}
