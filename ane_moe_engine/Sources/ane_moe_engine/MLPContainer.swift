//
//  MLPContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation

/// A hardware-accelerated execution container that drives ANE-optimized Conv2d SwiGLU MLP blocks using 3D tensor inputs.
public final class MLPContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let hiddenDimensions = 2048
    private let totalLayers: Int
    private var layers: [Int: MLModel] = [:]
    
    // MARK: - Initialization
    
    /// Initializes the MLP execution pipeline by loading static ANE-targeted graph networks.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 24) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            let mlpModelURL = layerDirectoryURL.appendingPathComponent("mlp_ane.mlmodelc")
            if fileManager.fileExists(atPath: mlpModelURL.path) {
                self.layers[layerIndex] = try await MLModel.load(contentsOf: mlpModelURL, configuration: configuration)
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the input sequence register straight through the ANE-optimized layers using a 3D tensor interface.
    ///
    /// - Parameters:
    ///   - hiddenStates: The input unified memory lane register currently housing the token features.
    ///   - layerIndex: The structural layer sequence identifier mapping to the targeted network graph node.
    /// - Returns: The newly projected output register containing computed MLP states.
    public func forward(_ hiddenStates: CVPixelBuffer, layerIndex: Int) async throws -> CVPixelBuffer {
        guard let mlpModel = layers[layerIndex] else { return hiddenStates }
        
        // 1. Lock pixel buffer and extract the raw hardware pointer
        CVPixelBufferLockBaseAddress(hiddenStates, [])
        defer { CVPixelBufferUnlockBaseAddress(hiddenStates, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(hiddenStates) else {
            throw NSError(domain: "MoEMlpPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to resolve hardware base address."])
        }
        
        // 2. Wrap the pixel buffer pointer in an MLMultiArray with a 3D tensor shape [1, 1, 2048] for zero-copy efficiency.
        // Assumes single-token generation (Tokens=1). For batching/multi-token sequences, pass the dimension dynamically.
        let shape: [NSNumber] = [1, 1, hiddenDimensions as NSNumber]
        let strides: [NSNumber] = [
            (1 * hiddenDimensions) as NSNumber,
            hiddenDimensions as NSNumber,
            1
        ]
        
        let inputMultiArray = try MLMultiArray(
            dataPointer: baseAddress,
            shape: shape,
            dataType: .float16, // Float16 layout matching kCVPixelFormatType_OneComponent16Half
            strides: strides,
            deallocator: { _ in }
        )
        
        // 3. Bind pure 3D tensor features straight to CoreML
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: inputMultiArray)
        ])
        
        // 4. Fire inference onto the hardware pipeline
        let prediction = try await mlpModel.prediction(from: inputs)
        
        // 5. Extract output tensor and map it back
        guard let outputFeature = prediction.featureValue(for: "down_out"),
              let outputMultiArray = outputFeature.multiArrayValue else {
            throw NSError(domain: "MoEMlpPipeline", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid tensor signature returned from graph."])
        }
        
        // TODO: Write the projected output back into the main stream (hiddenStates) if in-place modification is required.
        // Implement a fast memcpy or direct mapping from outputMultiArray to the targeted CVPixelBuffer if the model cannot overwrite in-place.
        
        return hiddenStates
    }
}
