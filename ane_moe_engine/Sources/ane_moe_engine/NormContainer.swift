//  NormContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A hardware-accelerated normalization manager that drives ANE-optimized LayerNorm-based RMSNorm blocks straight via Core AI.
public final class NormContainer: @unchecked Sendable {
  
  public let hiddenDimensions = 2048
  private let totalLayers: Int
  
  private var inputNormLayers: [Int: InferenceFunction] = [:]
  private var postAttentionNormModels: [Int: InferenceFunction] = [:]
  
  private var inputNormOutputs: [Int: NDArray] = [:]
  private var postAttentionNormOutputs: [Int: NDArray] = [:]
  
  public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
      self.totalLayers = totalLayers
      let fileManager = FileManager.default
      
      for layerIndex in 0..<totalLayers {
          let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
          guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
          
          // 1. Load input_layernorm
          let inputNormURL = layerDirectoryURL.appendingPathComponent("norm_input_layernorm_layer_\(layerIndex).aimodel")
          if fileManager.fileExists(atPath: inputNormURL.path) {
              let aiModel = try await AIModel(contentsOf: inputNormURL)
              if let function = try aiModel.loadFunction(named: "main") {
                  self.inputNormLayers[layerIndex] = function
                  self.inputNormOutputs[layerIndex] = NDArray(shape: [1, 1, 1, hiddenDimensions], scalarType: .float16)
              }
          }
          
          // 2. Load post_attention_layernorm
          let postNormURL = layerDirectoryURL.appendingPathComponent("norm_post_attention_layernorm_layer_\(layerIndex).aimodel")
          if fileManager.fileExists(atPath: postNormURL.path) {
              let aiModel = try await AIModel(contentsOf: postNormURL)
              if let function = try aiModel.loadFunction(named: "main") {
                  self.postAttentionNormModels[layerIndex] = function
                  self.postAttentionNormOutputs[layerIndex] = NDArray(shape: [1, 1, 1, hiddenDimensions], scalarType: .float16)
              }
          }
      }
  }
}


extension NormContainer {
    @discardableResult
    public func normalize(
        _ inputTensor: NDArray,
        layerIndex: Int,
        isPostAttention: Bool
    ) async throws -> NDArray {
        guard let function = isPostAttention ? postAttentionNormModels[layerIndex] : inputNormLayers[layerIndex],
              var outputTensor = isPostAttention ? postAttentionNormOutputs[layerIndex] : inputNormOutputs[layerIndex] else {
            return inputTensor
        }
        
        let inputs: [String: NDArray] = ["x": inputTensor]
        let outputKey = function.descriptor.outputNames.first ?? "down_out"
        
        var mutableViews = InferenceFunction.MutableViews()
        let mutableView = outputTensor.mutableView(as: Float16.self)
        mutableViews.insert(mutableView, for: outputKey)
        
        let emptyStates = InferenceFunction.MutableViews()
        
        _ = try await function.run(
            inputs: inputs,
            states: emptyStates,
            outputViews: mutableViews
        )
        
        return outputTensor
    }
}
