//
//  FullAttentionContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//
import CoreML
import Foundation


/// A hardware-accelerated state and weight manager that dynamically discovers and loads Softmax Attention blocks.
public final class FullAttentionContainer: @unchecked Sendable {
    private let totalLayers: Int
    
    // Core model architecture constants matching the Qwen3.5 Softmax Attention blueprint
    public let numHeads = 16
    public let numKVHeads = 2
    public let headDim = 256
    public let maxSequenceLength = 512
    public var kvCacheMatrixSize: Int { numKVHeads * headDim * maxSequenceLength } // 2 * 256 * 512 = 262,144
    
    private var layerModels: [Int: MLModel] = [:]
    private var stateRegistry: [Int: MLState] = [:]
    
    // Low-overhead CVPixelBuffer allocations dedicated to tracking current loop parameters
    private var keyPixelBuffers: [Int: CVPixelBuffer] = [:]
    private var valuePixelBuffers: [Int: CVPixelBuffer] = [:]
    
    // Shared parameters used as static references across execution blocks
    private let currentLengthArray: MLMultiArray
    private let cosArray: MLMultiArray
    private let sinArray: MLMultiArray
    
    /// Initializes the container by scanning the folder and allocating tracking hardware matrices.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 24) async throws {
        self.totalLayers = totalLayers
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        let pixelBufferAttributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        
        // Pre-allocate auxiliary inputs shared across attention layers
        self.currentLengthArray = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float32)
        self.cosArray = try MLMultiArray(shape: [1, 4096, 1, 1], dataType: .float32)
        self.sinArray = try MLMultiArray(shape: [1, 4096, 1, 1], dataType: .float32)
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let modelURL = folderContents?.first(where: { $0.pathExtension == "mlmodelc" }) else { continue }
            
            // Filter and extract traditional Softmax Attention blocks exclusively
            if modelURL.lastPathComponent.contains("softmax") || modelURL.lastPathComponent.contains("attention") {
                let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
                self.layerModels[layerIdx] = model
                
                // Allocate zero-copy registers for key caches
                var kBuffer: CVPixelBuffer? = nil
                let kStatus = CVPixelBufferCreate(
                    kCFAllocatorDefault, kvCacheMatrixSize, 1,
                    kCVPixelFormatType_OneComponent16Half, pixelBufferAttributes as CFDictionary, &kBuffer
                )
                
                // Allocate zero-copy registers for value caches
                var vBuffer: CVPixelBuffer? = nil
                let vStatus = CVPixelBufferCreate(
                    kCFAllocatorDefault, kvCacheMatrixSize, 1,
                    kCVPixelFormatType_OneComponent16Half, pixelBufferAttributes as CFDictionary, &vBuffer
                )
                
                guard kStatus == kCVReturnSuccess, let resolvedKBuffer = kBuffer,
                      vStatus == kCVReturnSuccess, let resolvedVBuffer = vBuffer else {
                    throw NSError(domain: "FullAttentionContainer", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate KV hardware memory allocation for layer \(layerIdx)"])
                }
                
                self.keyPixelBuffers[layerIdx] = resolvedKBuffer
                self.valuePixelBuffers[layerIdx] = resolvedVBuffer
                
                // Zero-initialize pre-allocated spatial ranges inside memory lanes
                CVPixelBufferLockBaseAddress(resolvedKBuffer, [])
                CVPixelBufferLockBaseAddress(resolvedVBuffer, [])
                let kPtr = CVPixelBufferGetBaseAddress(resolvedKBuffer)!.assumingMemoryBound(to: Float16.self)
                let vPtr = CVPixelBufferGetBaseAddress(resolvedVBuffer)!.assumingMemoryBound(to: Float16.self)
                memset(kPtr, 0, kvCacheMatrixSize * MemoryLayout<Float16>.stride)
                memset(vPtr, 0, kvCacheMatrixSize * MemoryLayout<Float16>.stride)
                CVPixelBufferUnlockBaseAddress(resolvedVBuffer, [])
                CVPixelBufferUnlockBaseAddress(resolvedKBuffer, [])
                
                // Secure modern framework registry states
                self.stateRegistry[layerIdx] = model.makeState()
            }
        }
    }
    
    /// Returns the compiled model instance containing the static weights for the specified layer.
    public func modelView(forLayer layerIdx: Int) -> MLModel? {
        return layerModels[layerIdx]
    }
    
    /// Returns the native hardware-backed state handle for the specified layer.
    public func stateView(forLayer layerIdx: Int) -> MLState? {
        return stateRegistry[layerIdx]
    }
    
    /// Generates structured operational dictionary blocks containing runtime tracking parameters.
    public func auxiliaryFeatures(forStep currentStep: Int) -> [String: MLFeatureValue] {
        // Hydrate current iteration context parameters
        let lengthPtr = currentLengthArray.dataPointer.assumingMemoryBound(to: Float.self)
        lengthPtr[0] = Float(currentStep)
        
        // Note: Real deployment requires pre-calculating or slicing the global RoPE matrix into cosArray/sinArray here
        
        return [
            "current_length": MLFeatureValue(multiArray: currentLengthArray),
            "cos": MLFeatureValue(multiArray: cosArray),
            "sin": MLFeatureValue(multiArray: sinArray)
        ]
    }
    
    /// Resets all hardware cache structures to zero across active memory slots.
    public func resetAllCaches() throws {
        for (layerIdx, model) in layerModels {
            if let kBuffer = keyPixelBuffers[layerIdx], let vBuffer = valuePixelBuffers[layerIdx] {
                CVPixelBufferLockBaseAddress(kBuffer, [])
                CVPixelBufferLockBaseAddress(vBuffer, [])
                let kPtr = CVPixelBufferGetBaseAddress(kBuffer)!.assumingMemoryBound(to: Float16.self)
                let vPtr = CVPixelBufferGetBaseAddress(vBuffer)!.assumingMemoryBound(to: Float16.self)
                memset(kPtr, 0, kvCacheMatrixSize * MemoryLayout<Float16>.stride)
                memset(vPtr, 0, kvCacheMatrixSize * MemoryLayout<Float16>.stride)
                CVPixelBufferUnlockBaseAddress(vBuffer, [])
                CVPixelBufferUnlockBaseAddress(kBuffer, [])
            }
            self.stateRegistry[layerIdx] = model.makeState()
        }
    }
}

