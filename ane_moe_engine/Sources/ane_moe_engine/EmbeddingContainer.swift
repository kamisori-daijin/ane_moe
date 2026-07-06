//
//  EmbeddingContainer.swift
//  ane_moe-Engine
//
//  Created by kamisori-daijin on 2026/06/04.
//

import CoreAI
import Foundation

/// Keeps the massive embedding weight matrix from `wte.bin` resident in memory
/// and extracts an ANE-optimized 4D tensor from any token ID using zero-copy.
public final class EmbeddingContainer: @unchecked Sendable {
    private let wteData: Data
    public let hiddenSize: Int
    
    /// Initializes the container with the URL of the `wte.bin` file on disk.
    public init(contentsOf wteURL: URL, hiddenSize: Int = 2048) throws {
        // Memory-map the file to leverage UMA (Unified Memory Architecture) without high memory overhead.
        self.wteData = try Data(contentsOf: wteURL, options: .mappedIfSafe)
        self.hiddenSize = hiddenSize
    }
    
    /// Performs an optimized lookup to create an `NDArray` from the token ID.
    @available(macOS 27.0, *)
    public func embeddingView(forTokenID tokenID: Int) -> NDArray? {
        let byteStride = hiddenSize * 2 // Float16 takes 2 bytes
        let tokenOffset = tokenID * byteStride
        
        // Bounds check
        guard tokenOffset + byteStride <= wteData.count else { return nil }
        
        // Match the 4D shape expected by the model to prevent validation errors
        let tensorShape = [1, hiddenSize, 1, 1]
        
        return wteData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> NDArray? in
            guard let baseAddress = rawBuffer.baseAddress else { return nil }
            let tokenPointer = baseAddress.advanced(by: tokenOffset)
            
            // 1. Instantiate the NDArray with the specified shape and scalar type
            var ndArray = NDArray(shape: tensorShape, scalarType: .float16)
            
            // 2. Get a typed MutableView using mutableView(as:) as specified in the documentation
            var mutableVectorView = ndArray.mutableView(as: Float16.self)
            
            // 3. Create a buffer pointer of hidden size elements from the memory-mapped pointer
            let typedPointer = tokenPointer.assumingMemoryBound(to: Float16.self)
            let bufferPointer = UnsafeBufferPointer(start: typedPointer, count: hiddenSize)
            
            // 4. Fast copy the elements in Row-Major order using copyElements(from:)
            mutableVectorView.copyElements(from: bufferPointer)
            
            return ndArray
        }
    }
}
