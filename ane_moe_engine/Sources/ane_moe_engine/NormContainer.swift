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
    
    // Dedicated destination pixel buffer pools to insulate execution states and prevent asynchronous race conditions.
    private var inputNormOutputs: [Int: CVPixelBuffer] = [:]
    private var postAttentionNormOutputs: [Int: CVPixelBuffer] = [:]
    
    // MARK: - Initialization
    
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        // Define explicit hardware-backed properties required for zero-copy CoreML and ANE destination binding.
        // Enforcing an empty dictionary on `kCVPixelBufferIOSurfacePropertiesKey` guarantees OS allocation on an IOSurface backing plane.
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        for layerIndex in 0..<totalLayers {
            let layerDirectoryURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIndex)")
            guard fileManager.fileExists(atPath: layerDirectoryURL.path) else { continue }
            
            // 1. Load input_layernorm models and allocate IOSurface registers
            let inputNormURL = layerDirectoryURL.appendingPathComponent("norm_input_layernorm_layer_\(layerIndex).mlmodelc")
            if fileManager.fileExists(atPath: inputNormURL.path) {
                self.inputNormLayers[layerIndex] = try await MLModel.load(contentsOf: inputNormURL, configuration: configuration)
                
                var outBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    hiddenDimensions, 1,
                    kCVPixelFormatType_OneComponent16Half,
                    bufferAttributes as CFDictionary, // Apply deterministic hardware attributes
                    &outBuffer
                )
                
                if status == kCVReturnSuccess, let resolvedBuffer = outBuffer {
                    self.inputNormOutputs[layerIndex] = resolvedBuffer
                } else {
                    throw NSError(domain: "NormContainer", code: -99, userInfo: [NSLocalizedDescriptionKey: "IOSurface creation failed for layer \(layerIndex) input norm"])
                }
            }
            
            // 2. Load post_attention_layernorm models and allocate IOSurface registers
            let postNormURL = layerDirectoryURL.appendingPathComponent("norm_post_attention_layernorm_layer_\(layerIndex).mlmodelc")
            if fileManager.fileExists(atPath: postNormURL.path) {
                self.postAttentionNormModels[layerIndex] = try await MLModel.load(contentsOf: postNormURL, configuration: configuration)
                
                var outBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    hiddenDimensions, 1,
                    kCVPixelFormatType_OneComponent16Half,
                    bufferAttributes as CFDictionary, // Apply deterministic hardware attributes
                    &outBuffer
                )
                
                if status == kCVReturnSuccess, let resolvedBuffer = outBuffer {
                    self.postAttentionNormOutputs[layerIndex] = resolvedBuffer
                } else {
                    throw NSError(domain: "NormContainer", code: -99, userInfo: [NSLocalizedDescriptionKey: "IOSurface creation failed for layer \(layerIndex) post norm"])
                }
            }
        }
    }
    
    // MARK: - Public Execution API
    
    @discardableResult
    public func normalize(_ buffer: CVPixelBuffer, layerIndex: Int, isPostAttention: Bool) async throws -> CVPixelBuffer {
        guard let normModel = isPostAttention ? postAttentionNormModels[layerIndex] : inputNormLayers[layerIndex],
              let outputRegister = isPostAttention ? postAttentionNormOutputs[layerIndex] : inputNormOutputs[layerIndex] else {
            return buffer
        }
        
        let tensorShape: [NSNumber] = [1, hiddenDimensions as NSNumber, 1, 1]
        
        // Wrap CVPixelBuffers straight into MLMultiArrays via native initializers to circumvent pointer arithmetic and eliminate type-inference bottlenecks.
        let inputMultiArray = try MLMultiArray(pixelBuffer: buffer, shape: tensorShape)
        let outputMultiArray = try MLMultiArray(pixelBuffer: outputRegister, shape: tensorShape)
        
        // Explicitly bind the source tensor to input port "x".
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: inputMultiArray)
        ])
        
        // Initialize inference options to explicitly enforce the destination backings array mapping.
        let options = MLPredictionOptions()
        let outputKey = normModel.modelDescription.outputDescriptionsByName.keys.first ?? "down_out"
        options.outputBackings = [
            outputKey: outputMultiArray
        ]
        
        // Dispatch inference utilizing the certified `prediction(from:options:)` signature rather than state-dependent providers.
        // This allows the ANE to securely execute hardware output redirection without race condition risk.
        _ = try await normModel.prediction(from: inputs, options: options)
        
        // Synchronize and overwrite computed results directly back into the primary token sequence stream.
        CVPixelBufferLockBaseAddress(buffer, [])
        CVPixelBufferLockBaseAddress(outputRegister, .readOnly)
        
        let dst = CVPixelBufferGetBaseAddress(buffer)!
        let src = CVPixelBufferGetBaseAddress(outputRegister)!
        memcpy(dst, src, 1 * hiddenDimensions * 1 * 1 * MemoryLayout<Float16>.stride)
        
        CVPixelBufferUnlockBaseAddress(outputRegister, .readOnly)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }
}
