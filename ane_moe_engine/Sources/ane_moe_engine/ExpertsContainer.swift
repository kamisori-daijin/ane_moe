//
//  ExpertsContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated execution container that dynamically loads, executes, and unloads 4-expert quantized slices.
public final class ExpertsContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let chunkSize = 4
    public let hiddenDimensions = 2048
    public let expertCount = 256
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private let baseDirectoryURL: URL
    private let modelConfiguration: MLModelConfiguration
    
    // Hardware attributes required for CoreML/ANE to accept the pixel buffers.
    private let bufferAttributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferCGImageCompatibilityKey: false,
        kCVPixelBufferCGBitmapContextCompatibilityKey: false
    ]
    
    // MARK: - Initialization
    
    /// Initializes the expert execution pipeline linked to targeted physical model paths.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) {
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
    ///   - sharedInputs: Unified input registry of MLMultiArrays using Anemll layout.
    ///   - sharedOutputs: Unified output registry of MLMultiArrays using Anemll layout.
    public func executeRequiredChunks(
        layerIndex: Int,
        activeExperts: Set<Int>,
        sharedInputs: [Int: MLMultiArray],
        sharedOutputs: [Int: MLMultiArray]
    ) async throws {
        let fileManager = FileManager.default
        
        // Element count for a 4-expert packed temporary buffer (4 * 1 * 2048 = 8192 elements)
        let chunkElements = chunkSize * 1 * hiddenDimensions
        
        // 100% matched with Python's dummy input shape: [current_chunk_size(4), tokens_per_expert(1), hidden_dim(2048)]
        let expertChunkShape: [NSNumber] = [chunkSize as NSNumber, 1, hiddenDimensions as NSNumber]
        
        // Byte size for a single expert slice (2048 * 2 = 4096 bytes)
        let singleExpertSliceBytes = 1 * hiddenDimensions * MemoryLayout<Float16>.stride
        
        // Isolate and unify target tracking starts inside unique set maps
        let requiredChunkStarts = Set(activeExperts.map { ($0 / chunkSize) * chunkSize })
        
        for startIndex in requiredChunkStarts {
            let chunkName = String(format: "qwen_expert_layer_%d_chunk_%03d.mlmodelc", layerIndex, startIndex)
            let chunkURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)").appendingPathComponent(chunkName)
            
            guard fileManager.fileExists(atPath: chunkURL.path) else { continue }
            
            // --- 1. Dynamic Model Load ---
            let chunkModel = try await MLModel.load(contentsOf: chunkURL, configuration: modelConfiguration)
            
            // Allocate IOSurface-backed workspace buffers with shape (Width = 8192, Height = 1) for Anemll alignment
            var inputWorkspace: CVPixelBuffer?
            var outputWorkspace: CVPixelBuffer?
            
            CVPixelBufferCreate(kCFAllocatorDefault, chunkElements, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &inputWorkspace)
            CVPixelBufferCreate(kCFAllocatorDefault, chunkElements, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &outputWorkspace)
            
            guard let resolvedInput = inputWorkspace, let resolvedOutput = outputWorkspace else { continue }
            
            // Create 3D MLMultiArrays directly from the pixel buffers with 100% accurate shape definition [4, 1, 2048]
            let inputChunkMultiArray = try MLMultiArray(pixelBuffer: resolvedInput, shape: expertChunkShape)
            let outputChunkMultiArray = try MLMultiArray(pixelBuffer: resolvedOutput, shape: expertChunkShape)
            
            // --- 2. Scatter Packing Phase ---
            // Pack individual active experts into a single, contiguous input workspace
            CVPixelBufferLockBaseAddress(resolvedInput, [])
            let destinationPointer = CVPixelBufferGetBaseAddress(resolvedInput)!.assumingMemoryBound(to: Float16.self)
            
            for offset in 0..<chunkSize {
                let globalExpertIndex = startIndex + offset
                
                if let individualInputTensor = sharedInputs[globalExpertIndex],
                   let individualPixelBuffer = individualInputTensor.pixelBuffer {
                    
                    CVPixelBufferLockBaseAddress(individualPixelBuffer, .readOnly)
                    let sourcePointer = CVPixelBufferGetBaseAddress(individualPixelBuffer)!.assumingMemoryBound(to: Float16.self)
                    
                    let targetOffset = offset * 1 * hiddenDimensions
                    memcpy(destinationPointer.advanced(by: targetOffset), sourcePointer, singleExpertSliceBytes)
                    
                    CVPixelBufferUnlockBaseAddress(individualPixelBuffer, .readOnly)
                }
            }
            CVPixelBufferUnlockBaseAddress(resolvedInput, [])
            
            // --- 3. Hardware Computation Execution (Anemll Bounding) ---
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "expert_batched_hidden_states": MLFeatureValue(multiArray: inputChunkMultiArray)
            ])
            
            // Pin the output directly to the allocated outputChunkMultiArray buffer
            let options = MLPredictionOptions()
            let outputKey = chunkModel.modelDescription.outputDescriptionsByName.keys.first ?? "down_out"
            options.outputBackings = [
                outputKey: outputChunkMultiArray
            ]
            
            // ANE execution; prevents memory race conditions and EXC_BAD_ACCESS
            _ = try await chunkModel.prediction(from: inputFeatures, options: options)
            
            // --- 4. Gather Unpacking Phase ---
            // Unpack results from the temporary workspace back into individual expert buffers
            CVPixelBufferLockBaseAddress(resolvedOutput, .readOnly)
            let globalSourcePointer = CVPixelBufferGetBaseAddress(resolvedOutput)!.assumingMemoryBound(to: Float16.self)
            
            for offset in 0..<chunkSize {
                let globalExpertIndex = startIndex + offset
                
                if let individualOutputTensor = sharedOutputs[globalExpertIndex],
                   let individualPixelBuffer = individualOutputTensor.pixelBuffer {
                    
                    CVPixelBufferLockBaseAddress(individualPixelBuffer, [])
                    let individualDestinationPointer = CVPixelBufferGetBaseAddress(individualPixelBuffer)!.assumingMemoryBound(to: Float16.self)
                    
                    let sourceOffset = offset * 1 * hiddenDimensions
                    memcpy(individualDestinationPointer, globalSourcePointer.advanced(by: sourceOffset), singleExpertSliceBytes)
                    
                    CVPixelBufferUnlockBaseAddress(individualPixelBuffer, [])
                }
            }
            CVPixelBufferUnlockBaseAddress(resolvedOutput, .readOnly)
            
            // --- 5. Automatic Memory Purge ---
        }
    }
}
