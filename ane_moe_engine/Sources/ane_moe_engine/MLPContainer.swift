//
//  MLPContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated execution container that drives ANE-optimized Conv2d SwiGLU MLP blocks straight via hardware textures.
public final class MLPContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let hiddenDimensions = 2048
    private let totalLayers: Int
    private var layers: [Int: MLModel] = [:]
    
    // Isolation pool for output data to prevent race conditions (Anemll concept).
    private var mlpOutputs: [Int: MLMultiArray] = [:]
    
    // MARK: - Initialization
    
    /// Initializes the MLP execution pipeline by loading static ANE-targeted graph networks.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        // Hardware attributes required for CoreML/ANE to accept the pixel buffers.
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        // Shape definition to fully align with the python converter model: [1, 1, 2048]
        let tensorShape: [NSNumber] = [1, 1, hiddenDimensions as NSNumber]
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            let mlpModelURL = layerDirectoryURL.appendingPathComponent("mlp_ane.mlmodelc")
            if fileManager.fileExists(atPath: mlpModelURL.path) {
                self.layers[layerIndex] = try await MLModel.load(contentsOf: mlpModelURL, configuration: configuration)
                
                // Pre-allocate layer-specific output buffers with shape (Width = 2048, Height = 1) for Anemll alignment
                var outBuffer: CVPixelBuffer?
                CVPixelBufferCreate(kCFAllocatorDefault, hiddenDimensions, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &outBuffer)
                if let buf = outBuffer {
                    // Wrap the allocated IOSurface buffer inside a zero-copy MLMultiArray descriptor
                    self.mlpOutputs[layerIndex] = MLMultiArray(pixelBuffer: buf, shape: tensorShape)
                }
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the input sequence register straight through the ANE-optimized layers using a clean tensor interface.
    ///
    /// - Parameters:
    ///   - inputTensor: The active token hidden states sequence multi-array backing to process.
    ///   - layerIndex: The structural layer sequence identifier mapping to the targeted network graph node.
    ///   - options: The dynamic prediction options tracking memory output backings.
    /// - Returns: The newly projected output register containing computed MLP states.
    public func forward(
        _ inputTensor: MLMultiArray,
        layerIndex: Int,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        guard let mlpModel = layers[layerIndex],
              let outputTensor = mlpOutputs[layerIndex] else {
            return inputTensor
        }
        
        // Pass the incoming MLMultiArray directly to the input feature provider dictionary without casting
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: inputTensor)
        ])
        
        // Specify zero-copy direct writing to the pre-allocated output backings pool
        let outputKey = mlpModel.modelDescription.outputDescriptionsByName.keys.first ?? "down_out"
        options.outputBackings = [
            outputKey: outputTensor
        ]
        
        // Core ANE inference; dumps results directly into the designated outputTensor hardware layer
        _ = try await mlpModel.prediction(from: inputs, options: options)
        
        // Pass the filled outputTensor to the next layer to maintain a copy-free ping-pong buffer scheme
        return outputTensor
    }
}
