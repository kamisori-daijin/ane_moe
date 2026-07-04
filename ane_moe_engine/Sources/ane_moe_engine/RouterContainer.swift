//
//  RouterContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A hardware-accelerated routing manager that dispatches token sequences to specific experts based on gating probabilities.
public final class RouterContainer: @unchecked Sendable {
    
    public let expertCount = 256
    public let topK = 2
    public let hiddenDimensions = 2048
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private var layers: [Int: InferenceFunction] = [:]
    
    public let sharedExpertInputs: [Int: NDArray]
    public let sharedExpertOutputs: [Int: NDArray]
    
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        let fileManager = FileManager.default
        
        var inputs: [Int: NDArray] = [:]
        var outputs: [Int: NDArray] = [:]
        
        for expertIndex in 0..<expertCount {
            inputs[expertIndex] = NDArray(shape: [1, maxSequenceLength, hiddenDimensions], scalarType: .float16)
            outputs[expertIndex] = NDArray(shape: [1, maxSequenceLength, hiddenDimensions], scalarType: .float16)
        }
        
        self.sharedExpertInputs = inputs
        self.sharedExpertOutputs = outputs
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            let routerModelURL = layerDirectoryURL.appendingPathComponent("router.aimodel")
            if fileManager.fileExists(atPath: routerModelURL.path) {
                let aiModel = try await AIModel(contentsOf: routerModelURL)
                self.layers[layerIndex] = try aiModel.loadFunction(named: "main")
            }
        }
    }
}


extension RouterContainer {
    @discardableResult
    public func route(
        _ hiddenStates: inout NDArray,
        layerIndex: Int,
        activeTokenCount tokenCount: Int
    ) async throws -> (outputTensor: NDArray, activeExperts: Set<Int>) {
        guard let function = layers[layerIndex] else {
            return (hiddenStates, [])
        }
        
        let inputs: [String: NDArray] = ["hidden_states": hiddenStates]
        
     
        var scoresArray = NDArray(shape: [1, tokenCount, topK], scalarType: .float32)
        var indicesArray = NDArray(shape: [1, tokenCount, topK], scalarType: .int32)
        
        var mutableViews = InferenceFunction.MutableViews()
        mutableViews.insert(scoresArray.mutableView(as: Float.self), for: "router_scores")
        mutableViews.insert(indicesArray.mutableView(as: Int32.self), for: "router_indices")
        
        _ = try await function.run(inputs: inputs, states: InferenceFunction.MutableViews(), outputViews: mutableViews)
        
        let activeExperts = Set<Int>()
        
        
        return (hiddenStates, activeExperts)
    }
}
