//
//  FullAttentionContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//
import CoreAI
import Foundation

/// A hardware-accelerated state manager that leverages Core AI InferenceFunctions for lightning-fast zero-overhead KV Caching.
public final class FullAttentionContainer: Sendable {
    private let totalLayers: Int
    public let hiddenDimensions = 2048
    
    // Core model architecture constants matching the Qwen3.5 Softmax Attention blueprint
    public let numHeads = 16
    public let numKVHeads = 2
    public let headDim = 256
    public let maxSequenceLength = 512
    public var kvCacheMatrixSize: Int { numKVHeads * headDim * maxSequenceLength } // 2 * 256 * 512 = 262,144
    
    
    private let layerFunctions: [Int: InferenceFunction]
    private let attentionOutputs: [Int: NDArray]
    
    // Shared operational registers allocated to track runtime tracking parameters
    private let currentLengthArray: NDArray
    private let cosArray: NDArray
    private let sinArray: NDArray
    
    // MARK: - Initialization
    
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let fileManager = FileManager.default
     
        self.currentLengthArray = NDArray(shape:[1, 1, 1, 1], scalarType: .float32)
        self.cosArray = NDArray(shape:[1, 4096, 1, 1], scalarType: .float32)
        self.sinArray = NDArray(shape:[1, 4096, 1, 1], scalarType: .float32)
        
        var functions: [Int: InferenceFunction] = [:]
        var outputs: [Int: NDArray] = [:]
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let contents = folderContents else { continue }
            
           
            guard let modelURL = contents.first(where: {
                $0.pathExtension == "aimodel" && $0.lastPathComponent.lowercased().contains("softmax_attention")
            }) else { continue }
            
       
            let asset = try AIModelAsset(contentsOf: modelURL)
            
       
            let aiModel = try await AIModel(contentsOf: modelURL)
            let function = try aiModel.loadFunction(named: "main") 
            functions[layerIdx] = function
            
         
            outputs[layerIdx] = NDArray(shape: [1, 1, 1, hiddenDimensions], scalarType: .float32)
        }
        
        self.layerFunctions = functions
        self.attentionOutputs = outputs
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the attention layer by binding input/output backings and leveraging the internal hardware-native KV cache state.
    public func executeAttention(
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
        
        // InferenceFunction run(inputs:states:outputViews:)
        let outputs = try await function.run(
            inputs: inputs,
            states: states,
            outputViews: outputViews
        )
        
        return outputs
    }
    
    /// Generates structured operational dictionary blocks containing runtime tracking parameters.
    public func auxiliaryFeatures(forStep currentStep: Int) -> [String: NDArray] {

        let updatedLength = NDArray(scalars: [Float(currentStep)], shape:[1, 1, 1, 1])
        
        return [
            "current_length": updatedLength,
            "cos": cosArray,
            "sin": sinArray
        ]
    }
    
    public func functionView(forLayer layerIdx: Int) -> InferenceFunction? { layerFunctions[layerIdx] }
    public func outputView(forLayer layerIdx: Int) -> NDArray? { attentionOutputs[layerIdx] }
}
