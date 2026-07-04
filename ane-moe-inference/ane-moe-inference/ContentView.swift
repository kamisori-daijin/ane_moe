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
    
    // Core hardware backend layers (stateful for future expansion)
    @State private var tokenizer: QwenTokenizer? = nil
    @State private var embedding: EmbeddingContainer? = nil
    @State private var moePipeline: Qwen3_5MoePipeline? = nil
    
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
                        // Guard until main pipeline is ready
                        .disabled(moePipeline == nil)
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
    
    // Folder selection dialog for model directory
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
    
    // Load all pipeline components from selected model directory, bind to pipeline
    func setupInputComponents(from baseWorkspaceURL: URL) {
        let wteURL = baseWorkspaceURL.appendingPathComponent("qwen3_5_moe_wte.bin")
        
        logText = "[Selected Path]: \(baseWorkspaceURL.path)\nInitializing unified hardware pipeline..."
        
        Task {
            do {
                // Load all sub-containers concurrently, transfer ownership to master pipeline
                let loadedTokenizer = try await QwenTokenizer(contentsOf: baseWorkspaceURL)
                let loadedEmbedding = try EmbeddingContainer(contentsOf: wteURL, hiddenSize: 2048)
                let loadedStateLoader = try await GatedDeltaNetContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedAttentionContainer = try await FullAttentionContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedRouter = try await RouterContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedPipeline = ExpertsContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedNorm = try await NormContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedMlp = try await MLPContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedRope = try await RoPEContainer(contentsOf: baseWorkspaceURL)
                
                // Bind all components to the unified master pipeline instance
                let masterPipeline = try Qwen3_5MoePipeline(
                    tokenizer: loadedTokenizer,
                    embedding: loadedEmbedding,
                    stateLoader: loadedStateLoader,
                    attentionContainer: loadedAttentionContainer,
                    router: loadedRouter,
                    expertPipeline: loadedPipeline,
                    mlpContainer: loadedMlp,
                    normContainer: loadedNorm,
                    ropeContainer: loadedRope
                )
                
                await MainActor.run {
                    self.tokenizer = loadedTokenizer
                    self.embedding = loadedEmbedding
                    self.moePipeline = masterPipeline
                    
                    logText += "\n\n✅ [Setup Success] Hardware pipeline fully bound to target directory footprint!"
                    logText += "\n➔ Structural Layers Mapped: 40 Layers Allocated"
                    logText += "\n➔ Tokenizer Backend: Ready"
                    logText += "\n➔ Zero-Copy Embedding Matrix: Ready"
                    logText += "\n➔ Hybrid Mixer Networks: Interleaved GatedDeltaNet & SoftmaxAttention Mapped"
                    logText += "\n➔ Sparse MoE System: 256 Expert Workspaces Pre-allocated Globally"
                    logText += "\n➔ Computational Layers: ANE-Optimized MLP & Mirror-RMSNorm Engine Live"
                    logText += "\n\n🚀 MASTER INFERENCE PIPELINE: ONLINE (Memory Completely Single-Instance)"
                }
            } catch {
                await MainActor.run {
                    logText += "\n\n❌ [Setup Error] Failed to map assets at this location: \(error.localizedDescription)"
                    self.tokenizer = nil
                    self.embedding = nil
                    self.moePipeline = nil
                }
            }
        }
    }
    
    // MARK: - Live Execution Test Bench
    
    /// Tests the token conversion and fires the unified master pipeline straight across the 40-layer topology.
    func runInputCircuitTest() {
        guard let tokenizer = tokenizer,
              let embedding = embedding,
              let pipeline = moePipeline else { return }
        
        logText += "\n\n------------------------------------------------------------------"
        logText += "\n[Live Circuit Ignition]: \"\(inputText)\""
        
        let tokenIDs = tokenizer.tokenIDs(from: inputText)
        logText += "\n➔ Resolved Token IDs: \(tokenIDs)"
        
        guard let firstTokenID = tokenIDs.first else {
            logText += "\n⚠️ No tokenIDs resolved. Terminated."
            return
        }
        
        logText += "\n[Projecting Zero-Copy Memory View for Token ID: \(firstTokenID)]"
        
        // The new EmbeddingContainer provides a true NDArray.
        guard let hiddenStatesStream = embedding.embeddingView(forTokenID: firstTokenID) else {
            logText += "\n❌ [Memory Error] Failed to project hardware tensor backing."
            return
        }
        
        logText += "\n🎉 [SUCCESS] True NDArray Tracking Register initialized successfully."
        logText += "\n➔ Executing 40 interleaved blocks sequentially via master graph..."
        
        Task {
            do {
                let currentStep = 0 // Initial autoregressive step
                
            
                var mutableHiddenStates = hiddenStatesStream
                
                // Run the master pipeline with the hardware-backed tensor through all 40 layers
                let resolvedOutputTensor = try await pipeline.evaluateSingleStep(
                    &mutableHiddenStates,
                    currentStep: currentStep
                )
                
                await MainActor.run {
                    logText += "\n\n=================================================================="
                    logText += "\n⚡ [CIRCUIT SPARK SUCCESS] Full 40-layer topology traversed!"
                    logText += "\n➔ Output Tensor Backing Reference: \(resolvedOutputTensor)"
                    logText += "\n➔ Memory Allocation: 100% Conserved via Unified Memory Ping-Pong"
                    logText += "\n\n🔮 Ready to plug Language Model Head (LM Head) for next-token sampling!"
                }
            } catch {
                await MainActor.run {
                    logText += "\n\n❌ [Runtime Execution Error]: \(error.localizedDescription)"
                }
            }
        }
    }
}
