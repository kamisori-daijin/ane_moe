//
//  ContentView.swift
//  ane-moe-inference
//
//  Created by kamisori-daijin on 2026/06/04.
//

import SwiftUI
import CoreML
import ane_moe_engine

struct ContentView: View {
    @State private var logText: String = "Engine Status: Idle. Please select your 'compiled_model' directory."
    
    @State private var tokenizer: QwenTokenizer? = nil
    @State private var embedding: EmbeddingContainer? = nil
    
    @State private var inputText: String = "Apple Silicon"
    
    // Track the custom model directory URL selected by the user
    @State private var selectedFolderURL: URL? = nil

    var body: some View {
        VStack(spacing: 15) {
            Text("Qwen3.5-35B-A3B Input Circuit Test Bench")
                .font(.headline)
                .padding(.top)
            
            // Folder selection UI component using NSOpenPanel
            HStack(spacing: 12) {
                Button("📁 Select Model Folder...") {
                    selectModelDirectoryWithPanel()
                }
                
                Text(selectedFolderURL?.path ?? "No folder selected (Click left button)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(selectedFolderURL == nil ? .red : .gray)
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.black.opacity(0.03))
                    .cornerRadius(4)
            }
            .padding(.horizontal)
            
            HStack {
                TextField("Enter text prompt to resolve tokenIDs:", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button("Spark Circuit") {
                    runInputCircuitTest()
                }
                .disabled(tokenizer == nil || embedding == nil)
                .keyboardShortcut("R", modifiers: .command) // Cmd + R to run the test
            }
            .padding(.horizontal)

            ScrollView {
                Text(logText)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Color.black.opacity(0.05))
            .cornerRadius(8)
            .frame(height: 200)
        }
        .frame(width: 650, height: 380)
    }

    /// Displays the macOS standard Finder panel (NSOpenPanel) to choose the model directory.
    func selectModelDirectoryWithPanel() {
        let panel = NSOpenPanel()
        panel.title = "Choose your compiled_model Directory"
        panel.showsHiddenFiles = false
        panel.canChooseFiles = false       // Restrict selection to directories only
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            guard let url = panel.url else { return }
            self.selectedFolderURL = url
            
            // Instantly initialize input stream from the selected URL
            setupInputComponents(from: url)
        }
    }

    /// Initializes `wte.bin` and `tokenizer.json` from the provided custom workspace directory.
    func setupInputComponents(from baseWorkspaceURL: URL) {
        let wteURL = baseWorkspaceURL.appendingPathComponent("qwen3_5_moe_wte.bin")
        let modelFolderURL = baseWorkspaceURL
        
        logText = "[Selected Path]: \(baseWorkspaceURL.path)\nInitializing gatekeepers..."
        
        Task {
            do {
                let loadedTokenizer = try await QwenTokenizer(contentsOf: modelFolderURL)
                let loadedEmbedding = try EmbeddingContainer(contentsOf: wteURL, hiddenSize: 4096)
                
                await MainActor.run {
                    self.tokenizer = loadedTokenizer
                    self.embedding = loadedEmbedding
                    logText += "\n\n✅ [Setup Success] Input pipeline successfully bound to selected hardware footprint!"
                    logText += "\n➔ Tokenizer Backend: Ready"
                    logText += "\n➔ Zero-Copy Embedding Matrix: Ready"
                }
            } catch {
                await MainActor.run {
                    logText += "\n\n❌ [Setup Error] Failed to map assets at this location: \(error.localizedDescription)"
                    logText += "\n⚠️ Ensure 'tokenizer.json' and 'qwen3_5_moe_wte.bin' both reside inside the selected directory."
                    self.tokenizer = nil
                    self.embedding = nil
                }
            }
        }
    }

    /// Tests the end-to-end pipeline: Input String ➔ Token IDs ➔ Zero-copy 4D Tensor.
    func runInputCircuitTest() {
        guard let tokenizer = tokenizer, let embedding = embedding else { return }
        
        logText += "\n\n------------------------------------------------------------------"
        logText += "\n[Input String]: \"\(inputText)\""
        
        let tokenIDs = tokenizer.tokenIDs(from: inputText)
        logText += "\n➔ Resolved Token IDs: \(tokenIDs)"
        
        guard let firstTokenID = tokenIDs.first else {
            logText += "\n⚠️ No tokenIDs resolved. Terminated."
            return
        }
        
        logText += "\n[Projecting Zero-Copy Memory View for Token ID: \(firstTokenID)]"
        
        if let inputTensor = embedding.embeddingView(forTokenID: firstTokenID) {
            logText += "\n🎉 [SUCCESS] 4D MultiArray memory slice mapped instantly!"
            logText += "\n➔ Projected Tensor Shape: \(inputTensor.shape) (NCHW Alignment)"
            logText += "\n➔ Tensor DataType: \(inputTensor.dataType == .float16 ? "Float16 (CoreML Native)" : "Other")"
            
            let ptr = inputTensor.dataPointer.assumingMemoryBound(to: Float16.self)
            let sampleValues = (0..<4).map { String(format: "%.4f", Float(ptr[$0])) }
            logText += "\n➔ Raw Memory Activations [0..3]: \(sampleValues)"
        } else {
            logText += "\n❌ [Runtime Error] Zero-copy memory slicing collapsed data boundaries."
        }
    }
}
