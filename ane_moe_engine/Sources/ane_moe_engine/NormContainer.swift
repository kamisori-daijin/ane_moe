//
//  NormContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated normalization manager that drives ANE-optimized LayerNorm-based RMSNorm blocks straight via hardware textures.
public final class NormContainer: @unchecked Sendable {
    
    // MARK: - Architectural Constants
    
    public let hiddenDimensions = 2048
    private let totalLayers: Int
    
    // Model storage registries mapped directly to structural layer indices
    private var inputNormLayers: [Int: MLModel] = [:]
    private var postAttentionNormModels: [Int: MLModel] = [:]
    
    // Isolation pools for output data to prevent race conditions (Anemll concept).
    private var inputNormOutputs: [Int: MLMultiArray] = [:]
    private var postAttentionNormOutputs: [Int: MLMultiArray] = [:]
    
    // MARK: - Initialization
    
    /// Initializes the normalization container by scanning the directory and loading static ANE-targeted network graphs.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        // Define explicit hardware-backed properties required for zero-copy CoreML and ANE destination binding.
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        // Shape definition to fully align with the python converter model: [1, 2048, 1, 1]
        let tensorShape: [NSNumber] = [1, hiddenDimensions as NSNumber, 1, 1]
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            // 1. Load input_layernorm models and allocate IOSurface registers
            let inputNormURL = layerDirectoryURL.appendingPathComponent("norm_input_layernorm_layer_\(layerIndex).mlmodelc")
            if fileManager.fileExists(atPath: inputNormURL.path) {
                self.inputNormLayers[layerIndex] = try await MLModel.load(contentsOf: inputNormURL, configuration: configuration)
                
                var outBuffer: CVPixelBuffer?
                
                // 💡 Fix: Align pixel buffer dimensions to match the trailing dimensions of the 4D tensor ([..., Height=2048, Width=1])
                // Swap Width and Height parameters to pass hardware-native stride validations.
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    1,                // Width = 1 (Matches tensorShape last dimension)
                    hiddenDimensions, // Height = 2048 (Matches tensorShape third dimension)
                    kCVPixelFormatType_OneComponent16Half,
                    bufferAttributes as CFDictionary,
                    &outBuffer
                )
                
                if status == kCVReturnSuccess, let resolvedBuffer = outBuffer {
                    // Wrap the allocated IOSurface buffer inside a zero-copy MLMultiArray descriptor
                    self.inputNormOutputs[layerIndex] = MLMultiArray(pixelBuffer: resolvedBuffer, shape: tensorShape)
                } else {
                    throw NSError(domain: "NormContainer", code: -99, userInfo: [NSLocalizedDescriptionKey: "IOSurface creation failed for layer \(layerIndex) input norm"])
                }
            }
            
            // 2. Load post_attention_layernorm models and allocate IOSurface registers
            let postNormURL = layerDirectoryURL.appendingPathComponent("norm_post_attention_layernorm_layer_\(layerIndex).mlmodelc")
            if fileManager.fileExists(atPath: postNormURL.path) {
                self.postAttentionNormModels[layerIndex] = try await MLModel.load(contentsOf: postNormURL, configuration: configuration)
                
                var outBuffer: CVPixelBuffer?
                
                // 💡 Fix: Align pixel buffer dimensions to match the trailing dimensions of the 4D tensor ([..., Height=2048, Width=1])
                // Swap Width and Height parameters to pass hardware-native stride validations.
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    1,                // Width = 1
                    hiddenDimensions, // Height = 2048
                    kCVPixelFormatType_OneComponent16Half,
                    bufferAttributes as CFDictionary,
                    &outBuffer
                )
                
                if status == kCVReturnSuccess, let resolvedBuffer = outBuffer {
                    // Wrap the allocated IOSurface buffer inside a zero-copy MLMultiArray descriptor
                    self.postAttentionNormOutputs[layerIndex] = MLMultiArray(pixelBuffer: resolvedBuffer, shape: tensorShape)
                } else {
                    throw NSError(domain: "NormContainer", code: -99, userInfo: [NSLocalizedDescriptionKey: "IOSurface creation failed for layer \(layerIndex) post norm"])
                }
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the sequence register straight through the ANE-optimized LayerNorm graphs utilizing hardware options.
    ///
    /// - Parameters:
    ///   - inputTensor: The primary sequence hidden states multi-array backing currently housing the active features.
    ///   - layerIndex: The structural layer sequence identifier mapping to the targeted network graph node.
    ///   - isPostAttention: A toggle flag determining whether to pull the primary input norm or secondary block norm.
    ///   - options: The dynamic prediction options tracking memory output backings.
    /// - Returns: The newly projected output register containing computed Norm states.
    @discardableResult
    public func normalize(
        _ inputTensor: MLMultiArray,
        layerIndex: Int,
        isPostAttention: Bool,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        guard let normModel = isPostAttention ? postAttentionNormModels[layerIndex] : inputNormLayers[layerIndex],
              let outputTensor = isPostAttention ? postAttentionNormOutputs[layerIndex] : inputNormOutputs[layerIndex] else {
            return inputTensor
        }
        
        // Pass the incoming MLMultiArray directly to the input feature provider dictionary without casting
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: inputTensor)
        ])
        
        // Specify zero-copy direct writing to the pre-allocated output backings pool
        let outputKey = normModel.modelDescription.outputDescriptionsByName.keys.first ?? "down_out"
        options.outputBackings = [
            outputKey: outputTensor
        ]
        
        // Core ANE inference; dumps results directly into the designated outputTensor hardware layer
        _ = try await normModel.prediction(from: inputs, options: options)
        
        // Pass the filled outputTensor to the next layer to maintain a copy-free ping-pong buffer scheme
        return outputTensor
    }
}
