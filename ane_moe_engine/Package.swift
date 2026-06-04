// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ane_moe_engine",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "ane_moe_engine",
            targets: ["ane_moe_engine"]
        ),
    ],
    dependencies: [
       
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "ane_moe_engine",
            dependencies: [
             
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers")
                // .product(name: "Models", package: "swift-transformers"),
                // .product(name: "Generation", package: "swift-transformers")
            ]
        ),
        

    ],
    swiftLanguageModes: [.v6]
)
