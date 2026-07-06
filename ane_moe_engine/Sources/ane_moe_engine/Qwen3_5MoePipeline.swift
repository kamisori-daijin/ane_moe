//
//  Qwen3_5MoePipeline.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A unified, zero-copy autoregressive inference pipeline managing interleaved DeltaNet, Softmax Attention, and MoE Expert layers via Core AI NDArray tracks.
public final class Qwen3_5MoePipeline: @unchecked Sendable {
    
    public let totalLayers = 40
    public let hiddenDimensions = 2048
    private let layerTypes: [String]
    
    private let tokenizer: QwenTokenizer
    private let embedding: EmbeddingContainer
    private let stateLoader: GatedDeltaNetContainer
    private let attentionContainer: FullAttentionContainer
    private let router: RouterContainer
    private let expertPipeline: ExpertsContainer
    private let mlpContainer: MLPContainer
    private let normContainer: NormContainer
    private let ropeContainer: RoPEContainer
    
    private let residualTensor: NDArray
    
    public init(
        tokenizer: QwenTokenizer, embedding: EmbeddingContainer, stateLoader: GatedDeltaNetContainer,
        attentionContainer: FullAttentionContainer, router: RouterContainer, expertPipeline: ExpertsContainer,
        mlpContainer: MLPContainer, normContainer: NormContainer, ropeContainer: RoPEContainer
    ) throws {
        self.tokenizer = tokenizer; self.embedding = embedding; self.stateLoader = stateLoader
        self.attentionContainer = attentionContainer; self.router = router; self.expertPipeline = expertPipeline
        self.mlpContainer = mlpContainer; self.normContainer = normContainer; self.ropeContainer = ropeContainer
        
        var types = [String](repeating: "linear_attention", count: totalLayers)
        for layerIndex in stride(from: 3, to: totalLayers, by: 4) { types[layerIndex] = "full_attention" }
        self.layerTypes = types
        
        self.residualTensor = NDArray(shape: [1, 1, hiddenDimensions], scalarType: .float16)
    }
}

extension Qwen3_5MoePipeline {
    @discardableResult
    public func evaluateSingleStep(_ hiddenStates: inout NDArray, currentStep step: Int) async throws -> NDArray {
        for layerIndex in 0..<totalLayers {
            let layerType = layerTypes[layerIndex]
            
            // 1. Residual Backup
            var mutableResidual = residualTensor
            try copyTensor(from: &hiddenStates, to: &mutableResidual)
            
            // 2. Pre-Normalization
            let normInput = try await normContainer.normalize(hiddenStates, layerIndex: layerIndex, isPostAttention: false)
            
            // 3. Attention Branch
            if layerType == "linear_attention" {
                    hiddenStates = try await stateLoader.forward(normInput, layerIndex: layerIndex)
                } else {
                            let aux = attentionContainer.auxiliaryFeatures(forStep: step)
                            if var cos = aux["cos"], var sin = aux["sin"], let len = aux["current_length"] {
                                try await ropeContainer.computeRoPE(currentLengthBuffer: len, destinationCos: &cos, destinationSin: &sin)
                            }
                            
                    var attentionOut = try await attentionContainer.executeAttention(normInput, layerIndex: layerIndex)
                            
                           
                    hiddenStates = try reshapeAndCopy(&attentionOut, toShape: [1, hiddenDimensions, 1, 1])
                        }
            try accumulateResidual(into: &hiddenStates, source: &mutableResidual)
            
            // 4. FFN Residual Backup
            try copyTensor(from: &hiddenStates, to: &mutableResidual)
            
            // 5. Post-Normalization
            let normPost = try await normContainer.normalize(hiddenStates, layerIndex: layerIndex, isPostAttention: true)
            
            // 6. FFN Branch (MoE or MLP)
            if layerIndex % 4 == 0 {
                var mutableNormPost = normPost
                let routingResult = try await router.route(&mutableNormPost, layerIndex: layerIndex, activeTokenCount: 1)
                try await expertPipeline.executeRequiredChunks(layerIndex: layerIndex, activeExperts: routingResult.activeExperts, sharedInputs: router.sharedExpertInputs, sharedOutputs: router.sharedExpertOutputs)
                hiddenStates = routingResult.outputTensor
            } else {
                hiddenStates = try await mlpContainer.forward(normPost, layerIndex: layerIndex)
            }
            try accumulateResidual(into: &hiddenStates, source: &mutableResidual)
        }
        return hiddenStates
    }
    
    private func reshapeAndCopy(_ source: inout NDArray, toShape newShape: [Int]) throws -> NDArray {
        var newTensor = NDArray(shape: newShape, scalarType: .float32)
            
        _ = newTensor.mutableView(as: Float32.self)
        _ = source.mutableView(as: Float32.self)
            
            
            
            return newTensor
    }
    private func copyTensor(from source: inout NDArray, to destination: inout NDArray) throws {
        _ = destination.mutableView(as: Float16.self)
        _ = source.mutableView(as: Float16.self)
    }
    
    private func accumulateResidual(into destination: inout NDArray, source: inout NDArray) throws {
        _ = destination.mutableView(as: Float16.self)
        _ = source.mutableView(as: Float16.self)
    }
}
