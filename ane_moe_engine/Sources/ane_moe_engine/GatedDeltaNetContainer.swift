//
//  GatedDeltaNetContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//

import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated state manager that automatically discovers and loads compiled Gated Delta Net layers from disk.
public final class GatedDeltaNetContainer: @unchecked Sendable {
    private let totalLayers: Int
    public let hiddenDimensions = 2048 // Fixed to hidden dimension size for Qwen3.5 (2048)
    
    private var layerModels: [Int: MLModel] = [:]
    
    // Core state matrices (e.g., "s_matrix" from ct.StateType) are fully managed and hidden inside the ANE via MLState.
    // This eliminates manual buffer allocations or memory resetting on the Swift side.
    private var stateRegistry: [Int: MLState] = [:]
    
    // Isolation pool for output data to prevent race conditions (Anemll concept).
    private var deltaNetOutputs: [Int: MLMultiArray] = [:]
    
    // MARK: - Initialization
    
    /// Initializes the loader by scanning and dynamically binding compiled models found inside the directory.
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        
        // Hardware attributes required for CoreML/ANE to accept the pixel buffers.
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        // Shape definition to fully align with the python converter model: [Batch=1, Height=1, Width=2048]
        // This 3D tensor layout maps perfectly to the CVPixelBuffer dimensions (Width = 2048, Height = 1).
        let tensorShape: [NSNumber] = [1, 1, hiddenDimensions as NSNumber]
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let contents = folderContents else { continue }
            
            // Targeted scan for compiled packages containing the "deltanet" keyword
            guard let modelURL = contents.first(where: {
                $0.pathExtension == "mlmodelc" && $0.lastPathComponent.lowercased().contains("deltanet")
            }) else { continue }
            
            let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
            self.layerModels[layerIdx] = model
            
            // Allocate state matrix registers inside the hardware via makeState().
            self.stateRegistry[layerIdx] = model.makeState()
            
            // Pre-allocate layer-specific output buffers with shape (Width = 2048, Height = 1) for Anemll alignment
            var outBuffer: CVPixelBuffer?
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                hiddenDimensions,
                1,
                kCVPixelFormatType_OneComponent16Half,
                bufferAttributes as CFDictionary,
                &outBuffer
            )
            
            if let buf = outBuffer {
                // Wrap the allocated IOSurface buffer inside a zero-copy MLMultiArray descriptor
                self.deltaNetOutputs[layerIdx] = MLMultiArray(pixelBuffer: buf, shape: tensorShape)
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the Gated Delta Net layer by binding input/output backings and leveraging the internal hardware-native recurrent state.
    public func forward(
        _ inputTensor: MLMultiArray,
        layerIndex: Int,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        guard let model = layerModels[layerIndex],
              let outputTensor = deltaNetOutputs[layerIndex],
              let state = stateRegistry[layerIndex] else {
            return inputTensor
        }
        
        // Pass the incoming MLMultiArray directly to the input feature provider dictionary
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: inputTensor)
        ])
        
        // Specify zero-copy direct writing to the pre-allocated output backings pool
        let outputKey = model.modelDescription.outputDescriptionsByName.keys.first ?? "output_hidden_states"
        options.outputBackings = [
            outputKey: outputTensor
        ]
        
        // Drive the ANE by binding input features, MLState (recurrent state), and output backings concurrently.
        _ = try await model.prediction(from: inputs, using: state, options: options)
        
        return outputTensor
    }
    
    /// Resets all recurrent state data registers to zero without re-allocating hardware descriptors.
    public final func resetAllStates() {
        for (layerIdx, model) in layerModels {
            // Re-creating the state object instantly forces the OS to clear the allocated ANE memory region.
            self.stateRegistry[layerIdx] = model.makeState()
        }
    }
    
    public func modelView(forLayer layerIdx: Int) -> MLModel? { layerModels[layerIdx] }
    public func stateView(forLayer layerIdx: Int) -> MLState? { stateRegistry[layerIdx] }
}
