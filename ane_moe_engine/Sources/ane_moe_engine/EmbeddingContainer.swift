//
//  EmbeddingContainer.swift
//  ane_moe-Engine
//
//  Created by kamisori-daijin on 2026/06/04.
//

import CoreML
import Foundation

/// Keeps the massive embedding weight matrix from `wte.bin` resident in memory
/// and extracts an ANE-optimized 4D tensor from any token ID using zero-copy.
public final class EmbeddingContainer: @unchecked Sendable {
    private let wteData: Data
    public let hiddenSize: Int
    
    /// Initializes the container with the URL of the `wte.bin` file on disk.
    public init(contentsOf wteURL: URL, hiddenSize: Int = 4096) throws {
        // Memory-map the file to leverage UMA (Unified Memory Architecture) without high memory overhead.
        self.wteData = try Data(contentsOf: wteURL, options: .mappedIfSafe)
        self.hiddenSize = hiddenSize
    }
    
    /// Performs a zero-copy lookup to create an `MLMultiArray` directly from the token ID.
    public func embeddingView(forTokenID tokenID: Int) -> MLMultiArray? {
        let byteStride = hiddenSize * 2 // Float16 takes 2 bytes
        let tokenOffset = tokenID * byteStride
        
        // Bounds check
        guard tokenOffset + byteStride <= wteData.count else { return nil }
        
        return wteData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> MLMultiArray? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            
            // Get the precise memory address for the target token
            let tokenPointer = baseAddress.advanced(by: tokenOffset)
            
            // Define 4D NCHW static alignment shape: [1, hidden_size, 1, 1]
            let shape: [NSNumber] = [1, NSNumber(value: hiddenSize), 1, 1]
            let strides: [NSNumber] = [NSNumber(value: hiddenSize), 1, 1, 1]
            
            // Wrap the raw pointer without copying data or allocating new memory
            return try? MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: tokenPointer),
                shape: shape,
                dataType: .float16, // Matches the expected input for ANE input_layernorm
                strides: strides,
                deallocator: { _ in } // wteData manages the lifetime; no deallocation needed here
            )
        }
    }
}
