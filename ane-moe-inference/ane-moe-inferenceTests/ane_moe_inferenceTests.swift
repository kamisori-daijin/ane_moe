//
//  ane_moe_inferenceTests.swift
//  ane-moe-inferenceTests
//
//  Created by kamisori-daijin on 2026/07/06.
//

import XCTest
import AppKit
@testable import ane_moe_engine
import CoreAI

final class PipelineTests: XCTestCase {
    
    @MainActor
    private func selectModelDirectory() async throws -> URL {
        return try await withCheckedThrowingContinuation { continuation in
            let panel = NSOpenPanel()
            panel.title = "Choose your compiled_model Directory for Testing"
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            
            if panel.runModal() == .OK, let url = panel.url {
                continuation.resume(returning: url)
            } else {
                continuation.resume(throwing: NSError(domain: "Test", code: -1, userInfo: [NSLocalizedDescriptionKey: "User cancelled selection"]))
            }
        }
    }
    
    func testFullPipelineStepByStep() async throws {
        let modelURL = try await selectModelDirectory()
        
        let tokenizer = try await QwenTokenizer(contentsOf: modelURL)
        let tokenIDs = tokenizer.tokenIDs(from: "Apple Silicon")
        
        let wteURL = modelURL.appendingPathComponent("qwen3_5_moe_wte.bin")
        let embedding = try EmbeddingContainer(contentsOf: wteURL, hiddenSize: 2048)
        
        guard let inputTensor = embedding.embeddingView(forTokenID: tokenIDs.first!) else { return }
        print("✅ Embedding Shape: \(inputTensor.shape)")
        
        let normContainer = try await NormContainer(contentsOf: modelURL, totalLayers: 40)
       
       
        let normOutput = try await normContainer.normalize(inputTensor, layerIndex: 0, isPostAttention: false)
        print("✅ Norm Output Shape: \(normOutput.shape)")
    }
}
