//
//  RoPEContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A hardware-accelerated matrix manager that drives ANE-optimized 4D-Native Rotary Embedding (RoPE) networks.
public final class RoPEContainer: @unchecked Sendable {
    
    public let headDimensions = 256
    public let numHeads = 16
    public let totalChannels = 4096
    
    private var ropeFunction: InferenceFunction?
    
    public init(contentsOf baseDirectoryURL: URL) async throws {
        let fileManager = FileManager.default
        let ropeModelURL = baseDirectoryURL.appendingPathComponent("qwen3_5_moe_rope.aimodel")
        
        if fileManager.fileExists(atPath: ropeModelURL.path) {
            let aiModel = try await AIModel(contentsOf: ropeModelURL)
            self.ropeFunction = try aiModel.loadFunction(named: "main")
        }
    }
}

extension RoPEContainer {
  
    public func computeRoPE(
        currentLengthBuffer: NDArray,
        destinationCos cosBuffer: inout NDArray,
        destinationSin sinBuffer: inout NDArray
    ) async throws {
        guard let function = ropeFunction else { return }
        
        let inputs: [String: NDArray] = ["current_length": currentLengthBuffer]
        
        var mutableViews = InferenceFunction.MutableViews()
        
       
        let cosView = cosBuffer.mutableView(as: Float16.self)
        mutableViews.insert(cosView, for: "cos_out")
        
        let sinView = sinBuffer.mutableView(as: Float16.self)
        mutableViews.insert(sinView, for: "sin_out")
        
        let emptyStates = InferenceFunction.MutableViews()
        
        _ = try await function.run(
            inputs: inputs,
            states: emptyStates,
            outputViews: mutableViews
        )
    }
}
