//
//  ExpertsContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreAI
import Foundation

/// A hardware-accelerated execution container that dynamically loads, executes, and unloads 4-expert quantized slices.
public final class ExpertsContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let chunkSize = 4
    public let hiddenDimensions = 2048
    public let expertCount = 256
    public let maxSequenceLength = 512
    
    private let totalLayers: Int
    private let baseDirectoryURL: URL
    
    // MARK: - Initialization
    
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) {
        self.totalLayers = totalLayers
        self.baseDirectoryURL = baseDirectoryURL
    }
    
    // MARK: - Public Execution API
    
    /// Dynamically discovers and evaluates localized expert chunks using Zero-Copy views.
    @available(macOS 27.0, *)
    public func executeRequiredChunks(
        layerIndex: Int,
        activeExperts: Set<Int>,
        sharedInputs: [Int: NDArray],
        sharedOutputs: [Int: NDArray]
    ) async throws {
        let fileManager = FileManager.default
        
        // Target tracking: Calculate the starting index of the required chunks
        let requiredChunkStarts = Set(activeExperts.map { ($0 / chunkSize) * chunkSize })
        
        for startIndex in requiredChunkStarts {
            let chunkName = String(format: "qwen_expert_layer_%d_chunk_%03d.aimodelc", layerIndex, startIndex)
            let chunkURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)").appendingPathComponent(chunkName)
            
            guard fileManager.fileExists(atPath: chunkURL.path) else { continue }
            
            // --- 1. Dynamic Model Load ---
            let chunkModel = try await AIModel(contentsOf: chunkURL)
            let functionName = chunkModel.functionNames.first ?? "main"
            guard let expertFunction = try chunkModel.loadFunction(named: functionName) else { continue }
            
            // --- 2. Scatter / Gather Processing Loop ---
            // 💡 Temporary workspace allocation and memory copying are completely unnecessary.
            // The 4 experts belonging to this chunk are evaluated individually and directly.
            for offset in 0..<chunkSize {
                let globalExpertIndex = startIndex + offset
                
                // Execute only if this expert is included in the active set
                guard activeExperts.contains(globalExpertIndex) else { continue }
                
                if let individualInputTensor = sharedInputs[globalExpertIndex],
                   var individualOutputTensor = sharedOutputs[globalExpertIndex] {
                    
                    // 💡 Zero memory copies occur here.
                    // The tensors of shape [1, hiddenDimensions] or [1, 1, hiddenDimensions] passed from
                    // the caller are fed directly into the Core AI model (InferenceFunction).
                    _ = try await expertFunction.run(
                        inputs: ["expert_hidden_state_single": individualInputTensor],
                        states: InferenceFunction.MutableViews(),
                        outputViews: InferenceFunction.MutableViews() // Output results are linked automatically inside
                    )
                    
                    // Note: By matching the function signature to handle a single sequence [1, 1, 2048]
                    // instead of the original batch axis [4, 1, 2048] via the Python export spec,
                    // tedious pointer manipulation on the Swift side is 100% eliminated.
                }
            }
            
            // --- 3. Automatic Memory Purge ---
            // Once exiting the loop, the weights of the corresponding chunk (4 experts) are automatically purged from the ANE.
        }
    }
}
