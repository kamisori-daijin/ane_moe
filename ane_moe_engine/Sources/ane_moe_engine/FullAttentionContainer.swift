//
//  FullAttentionContainer.swift
//  ane_moe_engine
//
//  Created by kamisori-daijin on 2026/06/05.
//
import CoreML
import Foundation
import CoreVideo

/// A hardware-accelerated state manager that leverages stateful MLState layers for lightning-fast zero-overhead KV Caching.
public final class FullAttentionContainer: @unchecked Sendable {
    private let totalLayers: Int
    public let hiddenDimensions = 2048
    
    // Core model architecture constants matching the Qwen3.5 Softmax Attention blueprint
    public let numHeads = 16
    public let numKVHeads = 2
    public let headDim = 256
    public let maxSequenceLength = 512
    public var kvCacheMatrixSize: Int { numKVHeads * headDim * maxSequenceLength } // 2 * 256 * 512 = 262,144
    
    private var layerModels: [Int: MLModel] = [:]
    
    // KV cache memory areas are fully managed and hidden inside the ANE via MLState.
    private var stateRegistry: [Int: MLState] = [:]
    
    // Isolation pool for output data to prevent race conditions (Anemll concept).
    private var attentionOutputs: [Int: MLMultiArray] = [:]
    
    // Shared operational registers allocated to track runtime tracking parameters
    private let currentLengthArray: MLMultiArray
    private let cosArray: MLMultiArray
    private let sinArray: MLMultiArray
    
    // MARK: - Initialization
    
    public init(contentsOf baseDirectoryURL: URL, totalLayers: Int = 40) async throws {
        self.totalLayers = totalLayers
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        let fileManager = FileManager.default
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        // Pre-allocate auxiliary helper tracking buffers
        self.currentLengthArray = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float32)
        self.cosArray = try MLMultiArray(shape: [1, 4096, 1, 1], dataType: .float32)
        self.sinArray = try MLMultiArray(shape: [1, 4096, 1, 1], dataType: .float32)
        
        let tensorShape: [NSNumber] = [1, 1, 1, hiddenDimensions as NSNumber]
        
        for layerIdx in 0..<totalLayers {
            let layerFolderURL = baseDirectoryURL.appendingPathComponent("layer_\(layerIdx)")
            guard fileManager.fileExists(atPath: layerFolderURL.path) else { continue }
            
            let folderContents = try? fileManager.contentsOfDirectory(at: layerFolderURL, includingPropertiesForKeys: nil)
            guard let contents = folderContents else { continue }
            
            guard let modelURL = contents.first(where: {
                $0.pathExtension == "mlmodelc" && $0.lastPathComponent.lowercased().contains("softmax_attention")
            }) else { continue }
            
            let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
            self.layerModels[layerIdx] = model
            
            // Allocate KV cache registers inside the hardware via makeState().
            self.stateRegistry[layerIdx] = model.makeState()
            
            // Pre-allocate layer-specific output buffers as MLMultiArrays.
            var outBuffer: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, hiddenDimensions, 1, kCVPixelFormatType_OneComponent16Half, bufferAttributes as CFDictionary, &outBuffer)
            if let buf = outBuffer {
                self.attentionOutputs[layerIdx] = MLMultiArray(pixelBuffer: buf, shape: tensorShape)
            }
        }
    }
    
    // MARK: - Public Execution API
    
    /// Evaluates the attention layer by binding input/output backings and leveraging the internal hardware-native KV cache state.
    public func executeAttention(
        _ inputTensor: MLMultiArray,
        layerIndex: Int,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        guard let model = layerModels[layerIndex],
              let outputTensor = attentionOutputs[layerIndex],
              let state = stateRegistry[layerIndex] else {
            return inputTensor
        }
        
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: inputTensor)
        ])
        
        let outputKey = model.modelDescription.outputDescriptionsByName.keys.first ?? "output_hidden_states"
        options.outputBackings = [
            outputKey: outputTensor
        ]
        
        _ = try await model.prediction(from: inputs, using: state, options: options)
        
        return outputTensor
    }
    
    /// Generates structured operational dictionary blocks containing runtime tracking parameters.
    public func auxiliaryFeatures(forStep currentStep: Int) -> [String: MLFeatureValue] {
        let lengthPtr = currentLengthArray.dataPointer.assumingMemoryBound(to: Float.self)
        lengthPtr[0] = Float(currentStep)
        
        return [
            "current_length": MLFeatureValue(multiArray: currentLengthArray),
            "cos": MLFeatureValue(multiArray: cosArray),
            "sin": MLFeatureValue(multiArray: sinArray)
        ]
    }
    
    /// Resets the internal KV cache state handles to zero across active slots straight inside the core registry.
    public final func resetAllCaches() {
        for (layerIdx, model) in layerModels {
            self.stateRegistry[layerIdx] = model.makeState()
        }
    }
    
    public func modelView(forLayer layerIdx: Int) -> MLModel? { layerModels[layerIdx] }
    public func stateView(forLayer layerIdx: Int) -> MLState? { stateRegistry[layerIdx] }
}
