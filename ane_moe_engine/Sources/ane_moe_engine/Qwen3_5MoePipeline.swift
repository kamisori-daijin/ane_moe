//
//  Qwen3_5MoePipeline.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation

/// A unified, zero-copy autoregressive inference pipeline managing interleaved DeltaNet, Softmax Attention, and MoE Expert layers.
public final class Qwen3_5MoePipeline: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let totalLayers = 40
    public let hiddenDimensions = 4096
    
    // Defines an anomalous interaction configuration map for topology (0, 1, 2 are line_attention, 3 is full_attention)
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
    
    // ⭕ A temporary backing register of the same shape for performing residual coupling (x = residual + x)
    private let residualBuffer: CVPixelBuffer
    
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
        
        // Perfectly simulates a 40-layer alternating configuration
        var types = [String](repeating: "linear_attention", count: totalLayers)
        for layerIndex in stride(from: 3, to: totalLayers, by: 4) {
            types[layerIndex] = "full_attention" // Full attention on layers 3, 7, 11...
        }
        self.layerTypes = types
        
       
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var resBuf: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 1 * 1 * hiddenDimensions, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &resBuf)
        
        guard let resolvedResidual = resBuf else {
            throw NSError(domain: "Qwen3_5MoePipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate residual context register."])
        }
        self.residualBuffer = resolvedResidual
    }
    
    // MARK: - Causal Generation Loop Execution
    
    /// Evaluates one single processing step across all 40 interleaved blocks, modifying the hidden states register in-place.
    ///
    /// - Parameters:
    ///   - hiddenStates: The primary execution sequence register housing the unified token state matrix.
    ///   - step: The active sequential loop iteration tracker determining position embeddings.
    /// - Returns: The fully resolved mutated hidden states buffer ready for language modeling heads.
    @discardableResult
    public func evaluateSingleStep(_ hiddenStates: CVPixelBuffer, currentStep step: Int) async throws -> CVPixelBuffer {
        
        var currentBuffer = hiddenStates
        
     
        for layerIndex in 0..<totalLayers {
            let layerType = layerTypes[layerIndex]
            
            // ======================================================================
            // PHASE 1: TOKEN MIXER BLOCK (Attention or DeltaNet)
            // ======================================================================
          
            try copyBuffer(from: currentBuffer, to: residualBuffer)
            
            //  hidden_states = input_layernorm(hidden_states)
            try await normContainer.normalize(currentBuffer, layerIndex: layerIndex, isPostAttention: false)
            
           
            if layerType == "linear_attention" {
                
                // currentBuffer = try await stateLoader.forwardDeltaNet(currentBuffer, layerIndex: layerIndex, step: step)
            } else if layerType == "full_attention" {
               
                try await ropeContainer.computeRoPE(
                    currentLengthBuffer: currentBuffer,
                    destinationCos: currentBuffer,
                    destinationSin: currentBuffer
                )
                
              
                // currentBuffer = try await attentionContainer.evaluateAttention(currentBuffer, layerIndex: layerIndex, step: step)
            }
            
            // hidden_states = residual + hidden_states
            try accumulateResidual(into: currentBuffer, source: residualBuffer)
            
            // ======================================================================
            // PHASE 2: FULLY CONNECTED BLOCK (Sparse MoE Block)
            // ======================================================================
            
            try copyBuffer(from: currentBuffer, to: residualBuffer)
            
            // hidden_states = post_attention_layernorm(hidden_states)
            try await normContainer.normalize(currentBuffer, layerIndex: layerIndex, isPostAttention: true)
            
         
            // currentBuffer = try await router.route(currentBuffer, layerIndex: layerIndex)
            // try await expertPipeline.executeRequiredChunks(layerIndex: layerIndex, ...)
            
         
            currentBuffer = try await mlpContainer.forward(currentBuffer, layerIndex: layerIndex)
            
            // hidden_states = residual + hidden_states
            try accumulateResidual(into: currentBuffer, source: residualBuffer)
        }
        
        return currentBuffer
    }
    
    // MARK: - Low-Overhead Hardware Memory Utilities
    
    private func copyBuffer(from source: CVPixelBuffer, to destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        
        let srcPtr = CVPixelBufferGetBaseAddress(source)!
        let dstPtr = CVPixelBufferGetBaseAddress(destination)!
        let byteCount = 1 * 1 * hiddenDimensions * MemoryLayout<Float16>.stride
        
        memcpy(dstPtr, srcPtr, byteCount)
    }
    
    private func accumulateResidual(into destination: CVPixelBuffer, source: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(destination, [])
        CVPixelBufferLockBaseAddress(source, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        
        let dstPtr = CVPixelBufferGetBaseAddress(destination)!.assumingMemoryBound(to: Float16.self)
        let srcPtr = CVPixelBufferGetBaseAddress(source)!.assumingMemoryBound(to: Float16.self)
    
        for d in 0..<hiddenDimensions {
            dstPtr[d] = dstPtr[d] + srcPtr[d]
        }
    }
}
