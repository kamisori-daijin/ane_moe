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
    @State private var stateLoader: GatedDeltaNetContainer? = nil
    // ⭕ Track the newly integrated traditional Softmax Attention container
    @State private var attentionContainer: FullAttentionContainer? = nil
    
    @State private var inputText: String = "Apple Silicon"
    @State private var selectedFolderURL: URL? = nil
    
 
    var body: some View {
        ZStack {
            
            
            VStack(spacing: 20) {
                Text("Qwen3.5-35B-A3B Input & Joint State Test Bench")
                    .font(.headline)
                    .padding(.top, 28)
                    .foregroundStyle(.primary)

                HStack(spacing: 12) {
                    Button(action: {
                        selectModelDirectoryWithPanel()
                    }) {
                        Label("Select Model Folder...", systemImage: "folder")
                    }
                    .buttonStyle(.bordered)
                    
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(selectedFolderURL?.path ?? "No folder selected (Click left button)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(selectedFolderURL == nil ? .red : .secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                    }
                
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .frame(height: 32)
                    .frame(maxWidth: .infinity)
                }
                .padding(.horizontal, 20)

                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        TextField("Enter text prompt to resolve tokenIDs:", text: $inputText)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 15, design: .monospaced))
                            .padding(.vertical, 4)
                            .frame(minWidth: 180, maxWidth: .infinity)

                        Button(action: {
                            runInputCircuitTest()
                        }) {
                            Text("Spark Circuit")
                                .bold()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(tokenizer == nil || embedding == nil || stateLoader == nil || attentionContainer == nil)
                        .keyboardShortcut("R", modifiers: .command)
                    }
                    .padding(.horizontal, 10)

                    ScrollView {
                        Text(logText)
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .foregroundStyle(.primary)
                            .multilineTextAlignment(.leading)
                    }
                    .frame(minHeight: 160, maxHeight: 240)
                    
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .padding(.horizontal, 2)
                }
                .padding(.vertical, 12)
                .padding(.horizontal, 22)
              
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .shadow(color: .black.opacity(0.1), radius: 16, y: 4)
                .frame(maxWidth: 720)

                Spacer(minLength: 12)
            }
            .padding(.vertical, 16)
        }
    }
  



    
    func selectModelDirectoryWithPanel() {
        let panel = NSOpenPanel()
        panel.title = "Choose your compiled_model Directory"
        panel.showsHiddenFiles = false
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK {
            guard let url = panel.url else { return }
            self.selectedFolderURL = url
            setupInputComponents(from: url)
        }
    }
    
    /// Initializes all tokenizers, embedding matrices, DeltaNet states, and GQA caches simultaneously from disk.
    func setupInputComponents(from baseWorkspaceURL: URL) {
        let wteURL = baseWorkspaceURL.appendingPathComponent("qwen3_5_moe_wte.bin")
        let modelFolderURL = baseWorkspaceURL
        
        logText = "[Selected Path]: \(baseWorkspaceURL.path)\nInitializing joint hardware registries..."
        
        Task {
            do {
                let loadedTokenizer = try await QwenTokenizer(contentsOf: modelFolderURL)
                let loadedEmbedding = try EmbeddingContainer(contentsOf: wteURL, hiddenSize: 4096)
                let loadedStateLoader = try await GatedDeltaNetContainer(contentsOf: baseWorkspaceURL, totalLayers: 24)
                // ⭕ Asynchronously auto-scan and load Softmax Attention blocks into their native registries
                let loadedAttentionContainer = try await FullAttentionContainer(contentsOf: baseWorkspaceURL, totalLayers: 24)
                
                await MainActor.run {
                    self.tokenizer = loadedTokenizer
                    self.embedding = loadedEmbedding
                    self.stateLoader = loadedStateLoader
                    self.attentionContainer = loadedAttentionContainer
                    
                    logText += "\n\n✅ [Setup Success] Hardware pipeline fully bound to target directory footprint!"
                    logText += "\n➔ Tokenizer Backend: Ready"
                    logText += "\n➔ Zero-Copy Embedding Matrix: Ready"
                    logText += "\n➔ GatedDeltaNet State Registries: Allocated and locked into Unified Memory"
                    logText += "\n➔ SoftmaxAttention KV Registries: Scanned and secured into target lanes"
                }
            } catch {
                await MainActor.run {
                    logText += "\n\n❌ [Setup Error] Failed to map assets at this location: \(error.localizedDescription)"
                    logText += "\n⚠️ Verify configuration files, embedding weights, and all layer compiled graphs exist inside this workspace."
                    self.tokenizer = nil
                    self.embedding = nil
                    self.stateLoader = nil
                    self.attentionContainer = nil
                }
            }
        }
    }
    
    /// Tests token conversion pipelines and prints state mapping statuses across interleaved architectures.
    func runInputCircuitTest() {
        guard let tokenizer = tokenizer,
              let embedding = embedding,
              let stateLoader = stateLoader,
              let attentionContainer = attentionContainer else { return }
        
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
            logText += "\n🎉 [SUCCESS] 4D MultiArray embedding slice mapped instantly!"
            logText += "\n➔ Projected Tensor Shape: \(inputTensor.shape) (NCHW Alignment)"
            
            // Inspect and sample Layer 0 vs Layer 1 to verify dynamic scanning and interleaved routing
            logText += "\n\n[Inspecting Joint Hardware Registries (Interleaved Topology Verification)]"
            
            // Layer 0: Traditional Softmax Attention Block
            if let layer0State = attentionContainer.stateView(forLayer: 0) {
                logText += "\n🔹 Layer 0 (Attention) -> Native MLState Object Verified: \(Unmanaged.passUnretained(layer0State).toOpaque())"
            }
            
            // Layer 1: Linear Recurrent Gated Delta Net Block
            if let layer1State = stateLoader.stateView(forLayer: 1),
               let backingArray = stateLoader.backingArrayView(forLayer: 1) {
                logText += "\n🔸 Layer 1 (DeltaNet)   -> Native MLState Object Verified: \(Unmanaged.passUnretained(layer1State).toOpaque())"
                logText += "\n   ➔ Packed Register Memory Map Shape: \(backingArray.shape)"
            }
        } else {
            logText += "\n❌ [Runtime Error] Zero-copy memory slicing collapsed data boundaries."
        }
    }
}

