//
//  RoPEContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/06.
//

import CoreML
import Foundation

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
    public func computeRoPE(
        currentLengthBuffer: CVPixelBuffer,
        destinationCos cosBuffer: CVPixelBuffer,
        destinationSin sinBuffer: CVPixelBuffer
    ) async throws {
        guard let model = ropeModel else { return }
        
     
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "current_length": MLFeatureValue(pixelBuffer: currentLengthBuffer)
        ])
        
      
        let prediction = try await model.prediction(from: inputs)
        
     
        
        guard let cosFeature = prediction.featureValue(for: "cos_out")?.imageBufferValue,
              let sinFeature = prediction.featureValue(for: "sin_out")?.imageBufferValue else {
            throw NSError(
                domain: "RotaryEmbeddingContainer",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to extract ANE-RoPE hardware matrices from the graph output."]
            )
        }
        
      
        CVPixelBufferLockBaseAddress(cosBuffer, [])
        CVPixelBufferLockBaseAddress(sinBuffer, [])
        CVPixelBufferLockBaseAddress(cosFeature, .readOnly)
        CVPixelBufferLockBaseAddress(sinFeature, .readOnly)
        
        let cosDst = CVPixelBufferGetBaseAddress(cosBuffer)!
        let sinDst = CVPixelBufferGetBaseAddress(sinBuffer)!
        let cosSrc = CVPixelBufferGetBaseAddress(cosFeature)!
        let sinSrc = CVPixelBufferGetBaseAddress(sinFeature)!
        
        let totalBytes = 1 * totalChannels * 1 * 1 * MemoryLayout<Float16>.stride // 4096 * 2 = 8192 bytes
        
        memcpy(cosDst, cosSrc, totalBytes)
        memcpy(sinDst, sinSrc, totalBytes)
        
        CVPixelBufferUnlockBaseAddress(sinFeature, .readOnly)
        CVPixelBufferUnlockBaseAddress(cosFeature, .readOnly)
        CVPixelBufferUnlockBaseAddress(sinBuffer, [])
        CVPixelBufferUnlockBaseAddress(cosBuffer, [])
    }
}
