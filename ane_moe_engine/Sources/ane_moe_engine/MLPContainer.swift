//
//  MLPContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A hardware-accelerated execution container that drives ANE-optimized SwiGLU MLP blocks straight via Core AI.
public final class MLPContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let hiddenDimensions = 2048
    private let totalLayers: Int
    private var layers: [Int: InferenceFunction] = [:]
    
    // Isolation pool for output data to prevent race conditions.
    private var mlpOutputs: [Int: NDArray] = [:]
    
    // MARK: - Initialization
    
    /// Initializes the MLP execution pipeline by loading static ANE-targeted graph networks.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let fileManager = FileManager.default
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            let mlpModelURL = layerDirectoryURL.appendingPathComponent("mlp_ane.aimodel")
            if fileManager.fileExists(atPath: mlpModelURL.path) {
                let aiModel = try await AIModel(contentsOf: mlpModelURL)
                if let function = try aiModel.loadFunction(named: "main") {
                    self.layers[layerIndex] = function
                    
                    // Pre-allocate layer-specific output buffers using NDArray
                    let outputArray = NDArray(shape: [1, 1, hiddenDimensions], scalarType: .float16)
                    self.mlpOutputs[layerIndex] = outputArray
                }
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the input sequence register straight through the ANE-optimized layers.
    public func forward(
        _ inputTensor: NDArray,
        layerIndex: Int
    ) async throws -> NDArray {
        guard let function = layers[layerIndex],
              let outputTensor = mlpOutputs[layerIndex] else {
            return inputTensor
        }
        
        let inputs: [String: NDArray] = ["hidden_states": inputTensor]
        let outputKey = function.descriptor.outputNames.first ?? "down_out"
        
        var mutableViews = InferenceFunction.MutableViews()
        
  
        var targetTensor = outputTensor
        let mutableView = targetTensor.mutableView(as: Float16.self)
        mutableViews.insert(mutableView, for: outputKey)
        
        let emptyStates = InferenceFunction.MutableViews()
        
   
        _ = try await function.run(
            inputs: inputs,
            states: emptyStates,
            outputViews: mutableViews
        )
       
        return targetTensor
    }
}
