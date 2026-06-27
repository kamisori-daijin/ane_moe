//
//  GatedDeltaNetContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//

import CoreAI
import Foundation

/// A hardware-accelerated state manager that automatically discovers and loads compiled Gated Delta Net layers from disk using Core AI.
public final class GatedDeltaNetContainer: Sendable {
    private let totalLayers: Int
    public let hiddenDimensions = 2048 // Fixed to hidden dimension size for Qwen3.5 (2048)
    
    // Core AI
    private let layerFunctions: [Int: InferenceFunction]
    private let deltaNetOutputs: [Int: NDArray]
    
    // MARK: - Initialization
    
    /// Initializes the loader by scanning and dynamically binding compiled models found inside the directory.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let fileManager = FileManager.default
        
        var functions: [Int: InferenceFunction] = [:]
        var outputs: [Int: NDArray] = [:]
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let contents = folderContents else { continue }
            
          
            guard let modelURL = contents.first(where: {
                $0.pathExtension == "aimodel" && $0.lastPathComponent.lowercased().contains("deltanet")
            }) else { continue }
            
         
            let aiModel = try await AIModel(contentsOf: modelURL)
            
         
            let function = try aiModel.loadFunction(named: "main")
            functions[layerIdx] = function
            

            outputs[layerIdx] = NDArray(shape: [1, 1, hiddenDimensions], scalarType: .float32)
        }
        
        self.layerFunctions = functions
        self.deltaNetOutputs = outputs
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the Gated Delta Net layer by binding input/output backings and leveraging the internal hardware-native recurrent state.
    /// - Parameters:
    ///   - inputTensor: The incoming hidden states as a Float32 NDArray.
    ///   - layerIndex: The index of the target Delta Net layer.
    ///   - states: The mutable state collection containing recurrent state matrices (s_matrix), passed as `consuming`.
    ///   - outputViews: The pre-allocated output backings to prevent allocation overhead, passed as `consuming`.
    public func forward(
        _ inputTensor: NDArray,
        layerIndex: Int,
        states: consuming InferenceFunction.MutableViews,
        outputViews: consuming InferenceFunction.MutableViews
    ) async throws -> InferenceFunction.Outputs {
        guard let function = layerFunctions[layerIndex] else {
            throw CocoaError(.fileNoSuchFile)
        }
        
     
        let inputs: [String: NDArray] = [
            "hidden_states": inputTensor
        ]
        
   
        let outputs = try await function.run(
            inputs: inputs,
            states: states,
            outputViews: outputViews
        )
        
        return outputs
    }
    
    public func functionView(forLayer layerIdx: Int) -> InferenceFunction? { layerFunctions[layerIdx] }
    public func outputView(forLayer layerIdx: Int) -> NDArray? { deltaNetOutputs[layerIdx] }
}
