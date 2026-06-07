//
//  RoPEContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated matrix manager that drives ANE-optimized 4D-Native Rotary Embedding (RoPE) networks.
public final class RoPEContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let headDimensions = 256
    public let numHeads = 16
    public let totalChannels = 4096 // 256 * 16 = 4096
    
    private var ropeModel: MLModel?
    
    // MARK: - Initialization
    
    /// Initializes the RoPE execution container by loading the static ANE-targeted generation graph.
    public init(contentsOf baseDirectoryURL: URL) async throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        let ropeModelURL = baseDirectoryURL.appendingPathComponent("qwen3_5_moe_rope.mlmodelc")
        
        if fileManager.fileExists(atPath: ropeModelURL.path) {
            self.ropeModel = try await MLModel.load(contentsOf: ropeModelURL, configuration: configuration)
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the current sequence index step straight into the ANE RoPE graph, hydrating the synchronized tracking registers.
    ///
    /// - Parameters:
    ///   - currentLengthBuffer: The 4D hardware register containing the active sequence execution index step.
    ///   - cosBuffer: The destination tracking register allocated to capture computed cosine frequencies.
    ///   - sinBuffer: The destination tracking register allocated to capture computed sine frequencies.
    ///   - options: The dynamic prediction options tracking memory output backings.
    public func computeRoPE(
        currentLengthBuffer: MLMultiArray,
        destinationCos cosBuffer: MLMultiArray,
        destinationSin sinBuffer: MLMultiArray,
        options: MLPredictionOptions
    ) async throws {
        guard let model = ropeModel else { return }
        
        // Pass the incoming MLMultiArray directly into the input feature provider node without casting
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "current_length": MLFeatureValue(multiArray: currentLengthBuffer)
        ])
        
        // Enforce zero-copy direct writing from ANE to pre-allocated target arrays via outputBackings
        // Aligns with Python multiple return keys: "cos_out" and "sin_out"
        options.outputBackings = [
            "cos_out": cosBuffer,
            "sin_out": sinBuffer
        ]
        
        // Core ANE inference; dumps frequency results directly into the designated buffers without race conditions
        _ = try await model.prediction(from: inputs, options: options)
        
        // Eliminates CPU locking overhead and memory copying entirely to preserve copy-free pipelining
    }
}
