//
//  QwenTokenizer.swift
//  ane_moe-Engine
//
//  Created by kamisori-daijin on 2026/06/04.
//

import Foundation
import Tokenizers
import Hub

/// A BPE tokenizer class that loads the vocabulary map from a local model directory
/// and handles bidirectional conversion between text and token IDs.
public final class QwenTokenizer: @unchecked Sendable {
    // Hold the abstract Tokenizer type from the swift-transformers library
    private let tokenizer: any Tokenizer
    
    /// Initializes the tokenizer from the parent directory containing the model configuration files.
    public init(contentsOf modelFolderURL: URL) async throws {
        // Instantiate the BPE environment using the asynchronous factory method
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)
        print("🎉 [Engine Tokenizer] Highly-optimized BPE backend instantiated via static signatures.")
    }
    
    /// Encodes human-readable text into an array of token IDs.
    public func tokenIDs(from text: String) -> [Int] {
        return tokenizer.encode(text: text)
    }
    
    /// Decodes an array of token IDs back into a UTF-8 string.
    public func string(fromTokenIDs tokenIDs: [Int]) -> String {
        return tokenizer.decode(tokens: tokenIDs)
    }
    
    /// Decodes a single token ID into its corresponding string representation.
    public func string(fromTokenID tokenID: Int) -> String {
        return tokenizer.decode(tokens: [tokenID])
    }
}
