//
//  GatedDeltaNetContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//

import CoreML
import Foundation

/// A hardware-accelerated state manager that automatically discovers and loads compiled Gated Delta Net layers from disk.
public final class GatedDeltaNetContainer: @unchecked Sendable {
    private let totalLayers: Int
    
    public let numValueHeads = 32
    public let headKeyDim = 128
    public let headValueDim = 128
    public var stateMatrixSize: Int { numValueHeads * headKeyDim * headValueDim }
    
    private var layerModels: [Int: MLModel] = [:]
    private var stateRegistry: [Int: MLState] = [:]
    
    private var statePixelBuffers: [Int: CVPixelBuffer] = [:]
    private var backingMultiArrays: [Int: MLMultiArray] = [:]
    
    /// Initializes the loader by scanning and dynamically binding compiled models found inside the directory.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 24) async throws {
        self.totalLayers = totalLayers
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        let pixelBufferAttributes: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        
        let shape: [NSNumber] = [1, NSNumber(value: numValueHeads * headKeyDim), 1, NSNumber(value: headValueDim)]
        let strides: [NSNumber] = [
            NSNumber(value: numValueHeads * headKeyDim * headValueDim),
            NSNumber(value: headValueDim),
            NSNumber(value: headValueDim),
            1
        ]
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            
            // Check if the directory for this specific layer exists on disk first
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            // Scan the folder contents dynamically to look for the compiled '.mlmodelc' package
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let modelURL = folderContents?.first(where: { $0.pathExtension == "mlmodelc" }) else {
                // If a layer folder exists but contains no compiled model, skip it safely
                continue
            }
            
            //  Detected a model, check if it's a Gated Delta Net model by inspecting the filename or loading it
            // (Since Gated Delta Net and Softmax Attention are interleaved, we filter by name or process dynamically)
            if modelURL.lastPathComponent.contains("deltanet") {
                let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
                self.layerModels[layerIdx] = model
                
                var pixelBuffer: CVPixelBuffer? = nil
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    stateMatrixSize,
                    1,
                    kCVPixelFormatType_OneComponent16Half,
                    pixelBufferAttributes as CFDictionary,
                    &pixelBuffer
                )
                
                guard status == kCVReturnSuccess, let resolvedBuffer = pixelBuffer else {
                    throw NSError(domain: "GatedDeltaNetStateLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate hardware state memory for layer \(layerIdx)"])
                }
                
                self.statePixelBuffers[layerIdx] = resolvedBuffer
                
                CVPixelBufferLockBaseAddress(resolvedBuffer, .readOnly)
                guard let rawBaseAddress = CVPixelBufferGetBaseAddress(resolvedBuffer) else {
                    CVPixelBufferUnlockBaseAddress(resolvedBuffer, .readOnly)
                    throw NSError(domain: "GatedDeltaNetStateLoader", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to retrieve base memory pointer for layer \(layerIdx)"])
                }
                CVPixelBufferUnlockBaseAddress(resolvedBuffer, .readOnly)
                
                let arrayView = try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(mutating: rawBaseAddress),
                    shape: shape,
                    dataType: .float16,
                    strides: strides,
                    deallocator: { _ in }
                )
                self.backingMultiArrays[layerIdx] = arrayView
                
                let ptr = rawBaseAddress.assumingMemoryBound(to: Float16.self)
                memset(ptr, 0, stateMatrixSize * MemoryLayout<Float16>.stride)
                
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
    
    /// Provides the zero-copy MLMultiArray view pointing directly to the underlying layer state pixel buffer.
    public func backingArrayView(forLayer layerIdx: Int) -> MLMultiArray? {
        return backingMultiArrays[layerIdx]
    }
    
    /// Resets all recurrent state data registers to zero without re-allocating hardware descriptors.
    public func resetAllStates() throws {
        for (layerIdx, model) in layerModels {
            if let pixelBuffer = statePixelBuffers[layerIdx] {
                CVPixelBufferLockBaseAddress(pixelBuffer, [])
                if let rawBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
                    let ptr = rawBaseAddress.assumingMemoryBound(to: Float16.self)
                    memset(ptr, 0, stateMatrixSize * MemoryLayout<Float16>.stride)
                }
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            }
            self.stateRegistry[layerIdx] = model.makeState()
        }
    }
}
