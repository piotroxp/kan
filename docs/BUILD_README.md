# KAN Speech Model

A speech recognition and generation model featuring quantum embeddings-based understanding core and Kolmogorov-Arnold Network (KAN) architecture.

## Features

- **KAN Layers**: Multiple KAN variants (B-spline, Chebyshev, Sinc, Fourier, RBF, Piecewise Linear)
- **Quantum Field Embeddings**: Squeezed coherent state wavefunctions for semantic representation
- **Modern C++20**: Clean, type-safe implementation
- **Comprehensive Tests**: Unit tests for all components

## Building

### Prerequisites

- C++20 compatible compiler (GCC 11+, Clang 14+)
- CMake 3.20+
- Conan 2.0+

### Build Steps

1. Install Conan dependencies:
```bash
conan install . --output-folder=build --build=missing
```

2. Build the project:
```bash
./build.sh
```

Or manually:
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

### Running Example

```bash
cd build
./example
```

## Project Structure

```
kan-speech-model/
├── src/
│   ├── core/           # KAN layer implementations
│   ├── quantum/        # Quantum wavefunction and embeddings
│   └── example/        # Example usage
├── tests/              # Unit tests
├── conanfile.txt       # Conan dependencies
└── CMakeLists.txt      # Build configuration
```

## Components

### KAN Layers

- **B-spline KAN**: Default safe choice, excellent convergence
- **Chebyshev KAN**: Smooth quantum state evolution
- **SincKAN**: High-frequency content, band-limited signals
- **Fourier KAN**: Periodic patterns
- **RBF KAN**: Fast, local patterns
- **Piecewise Linear KAN**: Speed-critical path

### Quantum Embeddings

Quantum field embeddings using squeezed coherent states:
- Wavefunction: ψ(x) = N exp[-(x-α)²/(4σ²) + iβx + iγ]
- Similarity: Born-rule fidelity |⟨ψ₁|ψ₂⟩|²
- Parameters extracted via Chebyshev KAN

## Research Context

This implementation is informed by two key research papers:

1. **A Practitioner's Guide to Kolmogorov–Arnold Networks** (arXiv:2510.25781v2) - Comprehensive review of KAN architectures, basis functions, and optimization strategies
2. **KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators** (arXiv:2512.23236v2) - Framework for GPU kernel optimization across heterogeneous hardware

See [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) for detailed analysis of how these papers inform the design decisions and implementation choices in this codebase.

## License

See LICENSE file for details.



