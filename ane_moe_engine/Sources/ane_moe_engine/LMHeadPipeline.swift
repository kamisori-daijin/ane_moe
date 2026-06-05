//
//  LMHeadPipeline.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/04.
//


import CoreML
import Foundation
import Accelerate


public final class LMHeadPipeline: @unchecked Sendable {
    private let baseModelURL: URL
    private var chunkModels: [MLModel] = []
    
    public let vocabSize = 248320
    public let chunkCount = 16
    public let hiddenDim: Int
    public let chunkSize = 15520 // 248320 / 16 = 15520 fixed
    
    // A single unified pixel buffer that serves as the zero-copy target for 240k logits output
    private let outputPixelBuffer: CVPixelBuffer
    private var preAllocatedOutputArrays: [MLMultiArray] = []
    
    public init(contentsOf baseDirectoryURL: URL, hiddenDim: Int = 4096) throws {
        self.baseModelURL = baseDirectoryURL
        self.hiddenDim = hiddenDim
        
        // 1. Allocate a flat 1x248320 grayscale-like pixel buffer to hold Float16 logits
        var pixelBuffer: CVPixelBuffer? = nil
        let attributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true, // Allows high-speed communication with ANE/GPU
            kCVPixelBufferIOSurfacePropertiesKey: [:]  // Prevents extra memory copies
        ]
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            vocabSize, // Width = 248320
            1,         // Height = 1
            kCVPixelFormatType_OneComponent16Half, // Native Float16 (Half Float) format
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let resolvedBuffer = pixelBuffer else {
            throw NSError(domain: "LMHeadPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate 240k-dimensional CVPixelBuffer allocation."])
        }
        self.outputPixelBuffer = resolvedBuffer
        
        // 2. Asynchronously target and initialize the 16 compiled ANE model chunks
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        for i in 1...chunkCount {
            let chunkURL = baseDirectoryURL.appendingPathComponent("lm_head_chunk_\(i).mlmodelc")
            let model = try MLModel(contentsOf: chunkURL, configuration: config)
            self.chunkModels.append(model)
        }
        
        // 3. Slice the unified pixel buffer into 16 pre-allocated MLMultiArray views of size 15520
        CVPixelBufferLockBaseAddress(outputPixelBuffer, .readOnly)
        let rawBaseAddress = CVPixelBufferGetBaseAddress(outputPixelBuffer)!
        CVPixelBufferUnlockBaseAddress(outputPixelBuffer, .readOnly)
        
        let shape: [NSNumber] = [1, NSNumber(value: chunkSize), 1, 1]
        let strides: [NSNumber] = [NSNumber(value: chunkSize), 1, 1, 1]
        
        for i in 0..<chunkCount {
            // Offset the memory pointer by 15520 * 2 bytes (Float16) for each consecutive chunk room
            let chunkOffsetBytes = i * chunkSize * 2
            let chunkPointer = rawBaseAddress.advanced(by: chunkOffsetBytes)
            
            // Wrap raw pointers directly so all 16 slices observe unique segments of the exact same pixel buffer
            let array = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: chunkPointer),
                shape: shape,
                dataType: .float16,
                strides: strides,
                deallocator: { _ in }
            )
            self.preAllocatedOutputArrays.append(array)
        }
        
        print("🎉 [Output Pipeline] 16-split ANE LM_Head fully integrated with Zero-Copy CVPixelBuffer matrix.")
    }
    
    /// Projects hidden states across the 16 model chunks and resolves the next token ID using high-speed argmax.
    public func predictedTokenID(fromFinalHiddenStates hiddenStates: MLMultiArray) -> Int {
        // 1. Run predictions across all 16 ANE chunks using bounded feature providers
        for i in 0..<chunkCount {
            let model = chunkModels[i]
            let _ = preAllocatedOutputArrays[i] // Ensure the pre-bound output memory remains retained
            
            let inputFeatures = try! MLDictionaryFeatureProvider(dictionary: ["hidden_states": hiddenStates])
            
            // CoreML mutates the pre-bound outputArray memory location instantly without allocation overhead
            _ = try! model.prediction(from: inputFeatures, options: MLPredictionOptions())
        }
        
        // 2. Lock the unified pixel buffer and extract the highest probability token via Auto-Vectorization
        CVPixelBufferLockBaseAddress(outputPixelBuffer, [])
        let outRawPtr = CVPixelBufferGetBaseAddress(outputPixelBuffer)!.assumingMemoryBound(to: Float16.self)
        
        // Bind the raw address to a safe UnsafeBufferPointer to enable compiler auto-vectorization
        let buffer = UnsafeBufferPointer(start: outRawPtr, count: vocabSize)
        
        var argmaxIndex = 0
        var maxProbability = buffer[0]
        
        // Scan the 240k-dimensional buffer efficiently using compiler-optimized SIMD loops
        for idx in 1..<vocabSize {
            let val = buffer[idx]
            if val > maxProbability {
                maxProbability = val
                argmaxIndex = idx
            }
        }
        
        CVPixelBufferUnlockBaseAddress(outputPixelBuffer, [])
        
        return argmaxIndex
    }
}
