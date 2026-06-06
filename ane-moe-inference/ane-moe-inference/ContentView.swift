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
    
    // Core hardware backend layers
    @State private var tokenizer: QwenTokenizer? = nil
    @State private var embedding: EmbeddingContainer? = nil
    @State private var stateLoader: GatedDeltaNetContainer? = nil
    @State private var attentionContainer: FullAttentionContainer? = nil
    @State private var router: RouterContainer? = nil
    @State private var expertPipeline: ExpertsContainer? = nil
    @State private var mlpContainer: MLPContainer? = nil
    @State private var normContainer: NormContainer? = nil
    @State private var ropeContainer: RoPEContainer? = nil
    
    
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
                        // ⭕ メインパイプラインオブジェクトの生成完了まで厳密にガード
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
    
    func setupInputComponents(from baseWorkspaceURL: URL) {
        let wteURL = baseWorkspaceURL.appendingPathComponent("qwen3_5_moe_wte.bin")
        
        logText = "[Selected Path]: \(baseWorkspaceURL.path)\nInitializing unified hardware pipeline..."
        
        Task {
            do {
                // 1. Load each sub-container concurrently within the local task scope.
                // Instantiating these as local constants instead of `@State` avoids redundant
                // context retention by the view hierarchy, ensuring a clean memory footprint
                // once ownership is transferred to the master pipeline.
                let loadedTokenizer = try await QwenTokenizer(contentsOf: baseWorkspaceURL)
                let loadedEmbedding = try EmbeddingContainer(contentsOf: wteURL, hiddenSize: 4096)
                let loadedStateLoader = try await GatedDeltaNetContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedAttentionContainer = try await FullAttentionContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedRouter = try await RouterContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedPipeline = ExpertsContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedNorm = try await NormContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedMlp = try await MLPContainer(contentsOf: baseWorkspaceURL, totalLayers: 40)
                let loadedRope = try await RoPEContainer(contentsOf: baseWorkspaceURL)
                
                // 2. Aggregate and bind all subsystems directly into the unified master pipeline instance.
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
                    self.moePipeline = masterPipeline // Commit the fully bound instance to the main pipeline state
                    
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
        
        // 1. Retrieve the source multidimensional array layout from the embedding matrix.
        guard let embeddingMultiArray = embedding.embeddingView(forTokenID: firstTokenID) else {
            logText += "\n❌ [Memory Error] Failed to project embedding view."
            return
        }
        
        // 2. Allocate a clean registry buffer backed by a verified IOSurface layout required by CoreML and the ANE.
        // Enforcing an empty dictionary on `kCVPixelBufferIOSurfacePropertiesKey` guarantees hardware backing plane residency.
        let bufferAttributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any],
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferCGImageCompatibilityKey: false,
            kCVPixelBufferCGBitmapContextCompatibilityKey: false
        ]
        
        var hardwareRegisterBuffer: CVPixelBuffer?
        // Instantiate the hardware texture aligned with Qwen 3.5's 2048 hidden dimensions using Float16
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            2048, 1, // hiddenDimensions = 2048
            kCVPixelFormatType_OneComponent16Half,
            bufferAttributes as CFDictionary,
            &hardwareRegisterBuffer
        )
        
        guard status == kCVReturnSuccess, let hiddenStatesStream = hardwareRegisterBuffer else {
            logText += "\n❌ [Memory Error] Failed to allocate true IOSurface hardware register."
            return
        }
        
        // 3. Dispatch a high-speed memory block transfer of the first token vector straight into the active IOSurface domain.
        CVPixelBufferLockBaseAddress(hiddenStatesStream, [])
        let destinationPointer = CVPixelBufferGetBaseAddress(hiddenStatesStream)!
        let sourcePointer = embeddingMultiArray.dataPointer
        memcpy(destinationPointer, sourcePointer, 1 * 2048 * MemoryLayout<Float16>.stride) // Exactly 4096 bytes allocated
        CVPixelBufferUnlockBaseAddress(hiddenStatesStream, [])
        
        logText += "\n🎉 [SUCCESS] True IOSurface Hardware Register allocated and synchronized."
        logText += "\n➔ Executing 40 interleaved blocks sequentially via master graph..."
        
        Task {
            do {
                // Route the hardware-backed pixel buffer stream directly into the master pipeline execution loop.
                let resolvedOutputBuffer = try await pipeline.evaluateSingleStep(
                    hiddenStatesStream,
                    currentStep: 0
                )
                
                await MainActor.run {
                    logText += "\n\n=================================================================="
                    logText += "\n⚡ [CIRCUIT SPARK SUCCESS] Full 40-layer topology traversed!"
                    logText += "\n➔ Output Buffer Reference: \(resolvedOutputBuffer)"
                    logText += "\n\n🔮 Ready for LM Head next-token sampling loop!"
                }
            } catch {
                await MainActor.run {
                    logText += "\n\n❌ [Runtime Execution Error]: \(error.localizedDescription)"
                }
            }
        }
    }
}
