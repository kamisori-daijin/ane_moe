//
//  LMHeadPipeline.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/04.
//


import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated causal language model head that projects final hidden states into a massive 151k vocabulary space across 16 ANE chunks.
public final class LMHeadPipeline: @unchecked Sendable {
    private let baseModelURL: URL
    private var chunkModels: [MLModel] = []
    
    // Aligned to padded vocabulary size matching 16 divisible ANE chunks (16 * 9504 = 152,064)
    public let vocabSize = 152064
    public let chunkCount = 16
    public let hiddenDim: Int
    public let chunkSize = 9504  // 152064 / 16 = 9504 fixed per ANE chunk
    
    // A single unified pixel buffer that serves as the zero-copy target for 152k logits output
    private let outputPixelBuffer: CVPixelBuffer
    private var preAllocatedOutputArrays: [MLMultiArray] = []
    
    // Reusable prediction options to eliminate allocation spikes
    private let predictionOptions = MLPredictionOptions()
    
    /// Initializes the LM Head pipeline by allocating the super-tensor IOSurface and loading the 16 model chunks.
    public init(contentsOf baseDirectoryURL: URL, hiddenDim: Int = 2048) throws {
        self.baseModelURL = baseDirectoryURL
        self.hiddenDim = hiddenDim
        
        // 1. Allocate a unified, super-tensor IOSurface pixel buffer for 152k logits
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        var pixelBuffer: CVPixelBuffer? = nil
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            vocabSize, // Width = 152064
            1,         // Height = 1
            kCVPixelFormatType_OneComponent16Half, // Native Float16
            bufferAttributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let resolvedBuffer = pixelBuffer else {
            throw NSError(domain: "LMHeadPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate true IOSurface for 152k logits matrix."])
        }
        self.outputPixelBuffer = resolvedBuffer
        
        // 2. Load all 16 compiled ANE chunk models
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        for i in 1...chunkCount {
            let chunkURL = baseDirectoryURL.appendingPathComponent("lm_head_chunk_\(i).mlmodelc")
            let model = try MLModel(contentsOf: chunkURL, configuration: config)
            self.chunkModels.append(model)
        }
        
        // 3. Slice the single IOSurface memory space into 16 distinct MLMultiArray view descriptors
        CVPixelBufferLockBaseAddress(outputPixelBuffer, .readOnly)
        guard let rawBaseAddress = CVPixelBufferGetBaseAddress(outputPixelBuffer) else {
            CVPixelBufferUnlockBaseAddress(outputPixelBuffer, .readOnly)
            throw NSError(domain: "LMHeadPipeline", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to retrieve hardware base address."])
        }
        CVPixelBufferUnlockBaseAddress(outputPixelBuffer, .readOnly)
        
        // Match the expected 4D output shape [1, chunkSize, 1, 1] of each python chunk model
        let chunkShape: [NSNumber] = [1, chunkSize as NSNumber, 1, 1]
        let chunkStrides: [NSNumber] = [chunkSize as NSNumber, 1, 1, 1]
        
        for i in 0..<chunkCount {
            // Calculate stride offsets (element count * 2 bytes for Float16)
            let chunkOffsetBytes = i * chunkSize * 2
            let chunkPointer = rawBaseAddress.advanced(by: chunkOffsetBytes)
            
            // Wrap the specific isolated pointer address inside a zero-copy MLMultiArray descriptor
            let sliceArray = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: chunkPointer),
                shape: chunkShape,
                dataType: .float16,
                strides: chunkStrides,
                deallocator: { _ in } // Lifecycle managed by the parent outputPixelBuffer
            )
            self.preAllocatedOutputArrays.append(sliceArray)
        }
        
        print("🎉 [Output Pipeline] 16-split ANE LM_Head fully integrated with Zero-Copy CVPixelBuffer matrix.")
    }
    
    // MARK: - Public Execution API
    
    /// Projects hidden states across the 16 model chunks and resolves the next token ID using high-speed argmax.
    ///
    /// - Parameter hiddenStates: The final token state tensor backing coming straight out of the 40-layer pipeline.
    /// - Returns: The factually resolved highest probability token ID integer.
    public func predictedTokenID(fromFinalHiddenStates hiddenStates: MLMultiArray) async throws -> Int {
        
        // 1. Execute inference sequentially across all 16 ANE chunk models
        for i in 0..<chunkCount {
            let model = chunkModels[i]
            let sliceTargetArray = preAllocatedOutputArrays[i]
            
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: hiddenStates)
            ])
            
            // Pin the execution output to the pre-allocated slice sliceTargetArray to prevent validation errors
            let outputKey = model.modelDescription.outputDescriptionsByName.keys.first ?? "logits"
            predictionOptions.outputBackings = [
                outputKey: sliceTargetArray
            ]
            
            // Core ANE inference; dumps results directly into the designated IOSurface slice
            _ = try await model.prediction(from: inputFeatures, options: predictionOptions)
        }
        
        // 2. Lock the giant IOSurface buffer to perform an auto-vectorized argmax loop on CPU
        CVPixelBufferLockBaseAddress(outputPixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(outputPixelBuffer, .readOnly) }
        
        guard let outRawPtr = CVPixelBufferGetBaseAddress(outputPixelBuffer) else {
            throw NSError(domain: "LMHeadPipeline", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to lock output buffer address for ArgMax scan."])
        }
        
        let typedPointer = outRawPtr.assumingMemoryBound(to: Float16.self)
        
        // Wrap in an unsafe buffer pointer to ensure compiler auto-vectorization (SIMD) optimizations
        let buffer = UnsafeBufferPointer(start: typedPointer, count: vocabSize)
        
        var argmaxIndex = 0
        var maxProbability = buffer[0]
        
        // Highly optimized scan across the 152k vocabulary dimension
        for idx in 1..<vocabSize {
            let val = buffer[idx]
            if val > maxProbability {
                maxProbability = val
                argmaxIndex = idx
            }
        }
        
        return argmaxIndex
    }
}
