//
//  EmbeddingContainer.swift
//  ane_moe-Engine
//
//  Created by kamisori-daijin on 2026/06/04.
//

import CoreML
import Foundation
import CoreVideo

/// Keeps the massive embedding weight matrix from `wte.bin` resident in memory
/// and extracts an ANE-optimized 4D tensor from any token ID using zero-copy.
public final class EmbeddingContainer: @unchecked Sendable {
    private let wteData: Data
    public let hiddenSize: Int
    
    // Hardware attributes required for CoreML/ANE to accept the pixel buffer.
    private let bufferAttributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any], // Force IOSurface backing
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferCGImageCompatibilityKey: false,
        kCVPixelBufferCGBitmapContextCompatibilityKey: false
    ]
    
    /// Initializes the container with the URL of the `wte.bin` file on disk.
    public init(contentsOf wteURL: URL, hiddenSize: Int = 4096) throws {
        // Memory-map the file to leverage UMA (Unified Memory Architecture) without high memory overhead.
        self.wteData = try Data(contentsOf: wteURL, options: .mappedIfSafe)
        self.hiddenSize = hiddenSize
    }
    
    /// Performs an optimized lookup to create an `MLMultiArray` backed by a true IOSurface from the token ID.
    public func embeddingView(forTokenID tokenID: Int) -> MLMultiArray? {
        let byteStride = hiddenSize * 2 // Float16 takes 2 bytes
        let tokenOffset = tokenID * byteStride
        
        // Bounds check
        guard tokenOffset + byteStride <= wteData.count else { return nil }
        
        var pixelBuffer: CVPixelBuffer?
        
        // 💡 Fix: Align pixel buffer dimensions to match the trailing dimensions of the 4D tensor ([1, 4096, Height=1, Width=1])
        // Swap Width and Height parameters to pass hardware-native stride validations.
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            1,          // Width = 1 (Matches tensorShape last dimension)
            hiddenSize, // Height = 4096 (Matches tensorShape second/third logic mapping)
            kCVPixelFormatType_OneComponent16Half,
            bufferAttributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let resolvedBuffer = pixelBuffer else { return nil }
        
        // Copy the token vector directly into the IOSurface memory address
        CVPixelBufferLockBaseAddress(resolvedBuffer, [])
        guard let destinationAddress = CVPixelBufferGetBaseAddress(resolvedBuffer) else {
            CVPixelBufferUnlockBaseAddress(resolvedBuffer, [])
            return nil
        }
        
        wteData.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> Void in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            let tokenPointer = baseAddress.advanced(by: tokenOffset)
            memcpy(destinationAddress, tokenPointer, byteStride)
        }
        CVPixelBufferUnlockBaseAddress(resolvedBuffer, [])
        
        // Match the 4D shape expected by the model to prevent validation errors
        let tensorShape: [NSNumber] = [1, hiddenSize as NSNumber, 1, 1]
        
        // Using MLMultiArray(pixelBuffer:shape:) ensures strict alignment and avoids layout conflicts.
        return try? MLMultiArray(pixelBuffer: resolvedBuffer, shape: tensorShape)
    }
}
