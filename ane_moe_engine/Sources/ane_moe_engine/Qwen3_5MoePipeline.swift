//
//  Qwen3_5MoePipeline.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation
import CoreVideo

/// A unified, zero-copy autoregressive inference pipeline managing interleaved DeltaNet, Softmax Attention, and MoE Expert layers via MLMultiArray IOSurface tracks.
public final class Qwen3_5MoePipeline: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let totalLayers = 40
    public let hiddenDimensions = 2048
    
    // Defines an interaction configuration map for topology (0, 1, 2 are linear_attention, 3 is full_attention)
    private let layerTypes: [String]
    
    // References to all containers that are initialized externally and passed from ContentView
    private let tokenizer: QwenTokenizer
    private let embedding: EmbeddingContainer
    private let stateLoader: GatedDeltaNetContainer
    private let attentionContainer: FullAttentionContainer
    private let router: RouterContainer
    private let expertPipeline: ExpertsContainer
    private let mlpContainer: MLPContainer
    private let normContainer: NormContainer
    private let ropeContainer: RoPEContainer
    
    // MLMultiArray backing with the same IOSurface lining for residual joining
    private let residualTensor: MLMultiArray
    private let reusable3DBuffer: MLMultiArray
    
    // A unified option for reuse across layers to completely eliminate allocation spikes.
    private let predictionOptions = MLPredictionOptions()
    
    // MARK: - Initialization
    
    /// Initializes the core causal model execution pipeline by syncing structural hardware subsystems.
    public init(
        tokenizer: QwenTokenizer,
        embedding: EmbeddingContainer,
        stateLoader: GatedDeltaNetContainer,
        attentionContainer: FullAttentionContainer,
        router: RouterContainer,
        expertPipeline: ExpertsContainer,
        mlpContainer: MLPContainer,
        normContainer: NormContainer,
        ropeContainer: RoPEContainer
    ) throws {
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.stateLoader = stateLoader
        self.attentionContainer = attentionContainer
        self.router = router
        self.expertPipeline = expertPipeline
        self.mlpContainer = mlpContainer
        self.normContainer = normContainer
        self.ropeContainer = ropeContainer
        
        // Perfectly simulates a 40-layer alternating configuration (0, 1, 2 are DeltaNet, 3 is Softmax Attention)
        var types = [String](repeating: "linear_attention", count: totalLayers)
        for layerIndex in stride(from: 3, to: totalLayers, by: 4) {
            types[layerIndex] = "full_attention" // Full attention on layers 3, 7, 11...
        }
        self.layerTypes = types
        
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        var resBuf: CVPixelBuffer?
        
        // Align pixel buffer dimensions to match the trailing dimensions of the 4D tensor ([1, 2048, Height=2048, Width=1])
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            1,                // Width = 1
            hiddenDimensions, // Height = 2048
            kCVPixelFormatType_OneComponent16Half,
            bufferAttributes as CFDictionary,
            &resBuf
        )

        
        
        guard let resolvedResidual = resBuf else {
            throw NSError(domain: "Qwen3_5MoePipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate residual context register."])
        }
        
        let tensorShape: [NSNumber] = [1, hiddenDimensions as NSNumber, 1, 1]
        self.residualTensor = try MLMultiArray(pixelBuffer: resolvedResidual, shape: tensorShape)
        let shape3D: [NSNumber] = [1, 1, hiddenDimensions as NSNumber]
        self.reusable3DBuffer = try MLMultiArray(shape: shape3D, dataType: .float16)
    }
    
    // MARK: - Causal Generation Loop Execution
    
    /// Evaluates one single processing step across all 40 interleaved blocks, modifying the hidden states register in-place.
    @discardableResult
    public func evaluateSingleStep(_ hiddenStates: MLMultiArray, currentStep step: Int) async throws -> MLMultiArray {
        var currentTensor = hiddenStates
        
        // Strict logical layouts mapped concurrently to clear hardware constraints
        let shape4D: [NSNumber] = [1, hiddenDimensions as NSNumber, 1, 1]
        let shape3D: [NSNumber] = [1, hiddenDimensions as NSNumber, 1]
        for layerIndex in 0..<totalLayers {
            let layerType = layerTypes[layerIndex]
            
            // Extract the true underlying hardware CVPixelBuffer token to pass constraints down the loop chain
            guard let underlyingBuffer = currentTensor.pixelBuffer else { return currentTensor }
            
            // 1. Back up residual data (Residual tracks are managed in 4D space)
            let current4D = try MLMultiArray(pixelBuffer: underlyingBuffer, shape: shape4D)
            try copyTensor(from: current4D, to: residualTensor)
            
            // 2. Execute pre-normalization layer (Takes 4D, returns 4D with native IOSurface backup)
            let normInput4D = try await normContainer.normalize(current4D, layerIndex: layerIndex, isPostAttention: false, options: predictionOptions)
            
            guard let normOutputBuffer = normInput4D.pixelBuffer else { continue }
            
            // 3. Transform 4D into 3D view preserving the original hardware-backed pixel buffer constraint
            let normInput3D = try MLMultiArray(pixelBuffer: normOutputBuffer, shape: shape3D)
            var branchOutput3D: MLMultiArray
            
            if layerType == "linear_attention" {
                // DeltaNet execution (Requires 3D input, returns 3D output)
                branchOutput3D = try await stateLoader.forward(normInput3D, layerIndex: layerIndex, options: predictionOptions)
            } else {
                // Softmax Attention execution
                let auxiliaryFeatures = attentionContainer.auxiliaryFeatures(forStep: step)
                if let cosTensor = auxiliaryFeatures["cos"]?.multiArrayValue,
                   let sinTensor = auxiliaryFeatures["sin"]?.multiArrayValue,
                   let currentLengthTensor = auxiliaryFeatures["current_length"]?.multiArrayValue {
                    
                    try await ropeContainer.computeRoPE(
                        currentLengthBuffer: currentLengthTensor,
                        destinationCos: cosTensor,
                        destinationSin: sinTensor,
                        options: predictionOptions
                    )
                }
                
                // Full Attention execution (Requires 3D input, returns 3D output)
                branchOutput3D = try await attentionContainer.executeAttention(normInput3D, layerIndex: layerIndex, options: predictionOptions)
            }
            
            // Transform branch output back to 4D for the next structural block joining
            guard let branchOutputBuffer = branchOutput3D.pixelBuffer else { continue }
            currentTensor = try MLMultiArray(pixelBuffer: branchOutputBuffer, shape: shape4D)
            try accumulateResidual(into: currentTensor, source: residualTensor)
            
            // --- FFN Layer Block ---
            guard let ffnUnderlyingBuffer = currentTensor.pixelBuffer else { continue }
            
            // 1. Evacuate residual data
            let ffnResidual4D = try MLMultiArray(pixelBuffer: ffnUnderlyingBuffer, shape: shape4D)
            try copyTensor(from: ffnResidual4D, to: residualTensor)
            
            // 2. Execute post-normalization layer (Takes 4D, returns 4D)
            let normPost4D = try await normContainer.normalize(currentTensor, layerIndex: layerIndex, isPostAttention: true, options: predictionOptions)
            
            guard let normPostOutputBuffer = normPost4D.pixelBuffer else { continue }
            
            // 3. Transform 4D post norm into 3D view for MoE routing block dispatching
            let normPost3D = try MLMultiArray(pixelBuffer: normPostOutputBuffer, shape: shape3D)
            var ffnOutput3D: MLMultiArray
            
            if layerIndex % 4 == 0 {
                // Router and Expert execution (Requires 3D input, returns 3D output)
                let routingResult = try await router.route(normPost3D, layerIndex: layerIndex, activeTokenCount: 1, options: predictionOptions)
                
                try await expertPipeline.executeRequiredChunks(
                    layerIndex: layerIndex,
                    activeExperts: routingResult.activeExperts,
                    sharedInputs: router.sharedExpertInputs,
                    sharedOutputs: router.sharedExpertOutputs
                )
                ffnOutput3D = routingResult.outputTensor
            } else {
                // Standard MLP execution (Requires 3D input, returns 3D output)
                ffnOutput3D = try await mlpContainer.forward(normPost3D, layerIndex: layerIndex, options: predictionOptions)
            }
            
            // Revert back to 4D to close the loop step block stream
            guard let ffnOutputBuffer = ffnOutput3D.pixelBuffer else { continue }
            currentTensor = try MLMultiArray(pixelBuffer: ffnOutputBuffer, shape: shape4D)
            try accumulateResidual(into: currentTensor, source: residualTensor)
        }
        
        return currentTensor
    }
    
    // MARK: - Low-Overhead Hardware Memory Utilities
    
    /// Performs a zero-copy fast memory copy from the source multi-array plane into the destination track.
    private func copyTensor(from source: MLMultiArray, to destination: MLMultiArray) throws {
        guard let srcBuf = source.pixelBuffer, let dstBuf = destination.pixelBuffer else { return }
        
        CVPixelBufferLockBaseAddress(srcBuf, .readOnly)
        CVPixelBufferLockBaseAddress(dstBuf, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(dstBuf, [])
            CVPixelBufferUnlockBaseAddress(srcBuf, .readOnly)
        }
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(srcBuf),
              let dstPtr = CVPixelBufferGetBaseAddress(dstBuf) else { return }
              
        let byteCount = 1 * hiddenDimensions * MemoryLayout<Float16>.stride
        memcpy(dstPtr, srcPtr, byteCount)
    }
    
    /// Performs an in-place element-wise Float16 addition across the hidden dimensions to compute residuals.
    private func accumulateResidual(into destination: MLMultiArray, source: MLMultiArray) throws {
        guard let dstBuf = destination.pixelBuffer, let srcBuf = source.pixelBuffer else { return }
        
        CVPixelBufferLockBaseAddress(dstBuf, [])
        CVPixelBufferLockBaseAddress(srcBuf, .readOnly)
        
        defer {
            CVPixelBufferUnlockBaseAddress(srcBuf, .readOnly)
            CVPixelBufferUnlockBaseAddress(dstBuf, [])
        }
        
        guard let rawDstPtr = CVPixelBufferGetBaseAddress(dstBuf),
              let rawSrcPtr = CVPixelBufferGetBaseAddress(srcBuf) else { return }
              
        let dstPtr = rawDstPtr.assumingMemoryBound(to: Float16.self)
        let srcPtr = rawSrcPtr.assumingMemoryBound(to: Float16.self)
        
        // Compiler auto-vectorization (SIMD) optimizes this sequential allocation loop straight on CPU
        for d in 0..<hiddenDimensions {
            dstPtr[d] = dstPtr[d] + srcPtr[d]
        }
    }
} // 💡 Here closes the public final class Qwen3_5MoePipeline boundary.
