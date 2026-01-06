# Research Context: PDF References for KAN Speech Model

This document provides context from two key research papers that inform the design and implementation of this KAN-based speech model with quantum embeddings.

---

## Papers Overview

### 1. A Practitioner's Guide to Kolmogorov–Arnold Networks
**arXiv: 2510.25781v2** | **Date: December 10, 2025**

A comprehensive review of KAN architectures, basis functions, optimization strategies, and practical guidance for selecting appropriate KAN variants.

### 2. KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
**arXiv: 2512.23236v2** | **Date: December 30, 2025**

A framework for automated kernel generation and optimization across heterogeneous AI accelerators (NVIDIA, AMD, MTIA), with emphasis on Triton, CuTe DSL, and low-level hardware abstractions.

---

## 1. KAN Architecture Context (2510.25781v2)

### 1.1 Basis Function Selection Rationale

The practitioner's guide emphasizes that **basis function choice is the primary inductive bias** in KAN architectures. Our implementation aligns with this principle:

| Component | KAN Type | Rationale (from paper + our design) |
|-----------|----------|-------------------------------------|
| **Audio Feature Extraction** | SincKAN | High-frequency content, band-limited signals, discontinuities in audio. Paper notes SincKAN > MLP for non-smooth functions. |
| **Quantum Embedding Encoder** | Chebyshev KAN | Smooth quantum state evolution, flat NTK spectrum, PDE-like properties. Paper shows Chebyshev KAN > MLP with faster convergence. |
| **Semantic Understanding** | B-spline KAN | Default safe choice, excellent convergence, compact support. Paper confirms B-spline as most widely used with proven stability. |
| **Language Modeling** | Fourier KAN | Periodic patterns in language, multi-scale temporal dependencies. Paper notes Fourier basis for periodic/oscillatory patterns. |
| **Attention Mechanism** | RBF KAN | Fast, local attention patterns, efficient similarity computation. Paper discusses RBF as fast spline alternative with local support. |
| **Output Generation** | Piecewise Linear KAN | Speed-critical path, ReLU-like efficiency. Paper notes piecewise linear for speed-critical applications. |

### 1.2 Key Insights from the Paper

#### Basis Function Characteristics

**B-spline KAN** (our default for semantic layers):
- Compact support: only 4 grid points active per evaluation
- C² smoothness (cubic B-splines)
- Excellent Sobolev/Besov convergence rates
- Most stable and widely adopted

**Chebyshev KAN** (our quantum embedding core):
- Flat NTK (Neural Tangent Kernel) spectrum → faster convergence
- Superior for PDE-like problems (quantum wavefunctions are PDE solutions)
- Global smoothness properties
- Recurrence relation: T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)

**SincKAN** (our audio feature extraction):
- Ideal for band-limited signals (audio is naturally band-limited)
- Handles discontinuities and sharp gradients
- Interpolation formula: φ(x) = Σ c_k sinc(π(x - kh)/h)

**Fourier KAN** (our language model):
- Periodic patterns in language (syntax, semantics, phonetics)
- Multi-scale temporal dependencies
- Spectral representation advantages

**RBF KAN** (our attention mechanism):
- Fast local similarity computation
- Trade-off: large ε → smooth but ill-conditioned, small ε → sharp and stable
- Efficient for attention patterns

**Piecewise Linear KAN** (our output generation):
- ReLU-like efficiency
- Speed-critical path optimization
- Minimal computational overhead

### 1.3 Accuracy and Convergence Insights

The paper's Table 2 shows performance trends:
- **KAN ≥ MLP** in most cases for accuracy
- **Faster convergence** (fewer epochs) but **slower training** (higher per-iteration cost)
- Basis-specific performance varies significantly with problem type
- Chebyshev KAN shows "faster convergence" for smooth/PDE problems
- SincKAN shows "faster convergence" for non-smooth problems

**Implications for our implementation:**
- Our multi-basis architecture (different KAN types per stage) aligns with paper's recommendation that optimal basis varies with problem characteristics
- Quantum embeddings (PDE-like) benefit from Chebyshev's flat NTK spectrum
- Audio processing (band-limited, potentially discontinuous) benefits from SincKAN
- Language modeling (periodic patterns) benefits from Fourier KAN

### 1.4 Efficiency Considerations

**Section 8: Efficiency Improvement** discusses:
- **Parallelism, GPU, and JAX Engineering**: Our ROCm/HIP GPU kernels align with this
- **Matrix Optimization & Efficient Bases**: Our basis-specific optimizations (compact B-spline support, RBF locality) follow these principles

### 1.5 Practical "Choose-Your-KAN" Guide (Section 11)

The paper provides a decision tree for KAN selection:
- **Smooth functions/PDEs** → Chebyshev (matches our quantum embeddings)
- **Discontinuities/sharp gradients** → Sinc (matches our audio processing)
- **Periodic patterns** → Fourier (matches our language model)
- **Speed-critical paths** → Piecewise Linear (matches our output generation)
- **Default/safe choice** → B-spline (matches our semantic layers)

**Our architecture follows this guidance precisely.**

---

## 2. GPU Kernel Optimization Context (2512.23236v2)

### 2.1 Heterogeneous Hardware Support

The KernelEvolve paper addresses optimization for:
- **NVIDIA GPUs** (CUDA, Triton)
- **AMD GPUs** (ROCm/HIP, Triton-AMD)
- **Meta MTIA** (custom accelerators)

**Relevance to our project:**
- We target **AMD Radeon RX 7900 XTX** (ROCm/HIP)
- The paper discusses Triton's multi-target compilation (Figure 2) supporting AMD via TritonAMDGPU-MLIR
- Our GPU kernels in `src/gpu/` should leverage Triton where possible for portability

### 2.2 Programming Model Fragmentation

The paper identifies fragmentation across:
- **CUDA** (thread-block model)
- **Triton** (tile-based DSL with automatic memory coalescing)
- **ROCm/HIP** (AMD extensions)
- **CuTe** (layout algebra for NVIDIA Hopper)
- **MTIA DSL** (C++ kernel DSL)

**Our current approach:**
- Using **HIP** for AMD GPU kernels (`src/gpu/hip_kernels.cu`)
- Consider **Triton** for future portability (supports AMD via TritonAMDGPU-MLIR)

### 2.3 Kernel Coverage Requirements

The paper emphasizes that **preprocessing operators are first-class optimization targets**, not just compute-intensive kernels:
- Missing preprocessing kernels force disaggregated architectures
- Architectural penalties (network latency, serialization) exceed individual kernel inefficiency
- Comprehensive kernel coverage enables monolithic accelerator deployment

**Implications:**
- Our audio preprocessing (mel-spectrogram, normalization) should have optimized GPU kernels
- Quantum embedding operations (wavefunction evaluation, fidelity computation) need GPU acceleration
- All KAN layer operations should have GPU implementations

### 2.4 Performance Targets

The paper reports:
- **17× performance improvements** over PyTorch baselines
- **100% correctness** across 480 operator-platform configurations
- **Weeks to hours** development time reduction

**Our goals:**
- Optimize KAN forward/backward passes on AMD GPU
- Ensure all model components have GPU implementations
- Target sub-100ms inference latency (aligned with paper's ads serving constraints)

### 2.5 Triton Multi-Target Architecture

Figure 2 shows Triton's compilation pipeline:
```
Triton Code → Triton-MLIR → TritonAMDGPU-MLIR → LLVM-IR → AMDGCN/HSACO
```

**Recommendation:**
- Consider migrating from HIP to Triton for:
  - Automatic memory coalescing
  - Multi-platform portability (NVIDIA, AMD, future accelerators)
  - Higher-level abstractions reducing kernel development time

---

## 3. Integration Points

### 3.1 KAN Basis Function Implementation

Our implementations in `src/core/` align with paper recommendations:

**B-spline KAN** (`bspline_kan.hpp`):
- Cubic B-splines (degree 3) with compact support
- Grid-based evaluation matching paper's formulation

**Chebyshev KAN** (`chebyshev_kan.hpp`):
- Recurrence relation implementation
- tanh(x) input transformation (matches paper's formulation)

**SincKAN** (`sinc_kan.hpp`):
- Sinc interpolation with spacing h = π/5
- Band-limited signal handling

**Fourier KAN** (`fourier_kan.hpp`):
- Periodic basis functions for language patterns

**RBF KAN** (`rbf_kan.hpp`):
- Gaussian RBF with configurable ε parameter

**Piecewise Linear KAN** (`piecewise_linear_kan.hpp`):
- ReLU-like efficiency for output generation

### 3.2 GPU Kernel Development

Current state (`src/gpu/`):
- HIP kernels for AMD GPU
- Basic KAN operations

**Future enhancements based on KernelEvolve:**
1. **Triton migration**: Use Triton for automatic optimization and portability
2. **Kernel fusion**: Fuse KAN operations (e.g., Chebyshev → B-spline in quantum→semantic path)
3. **Preprocessing kernels**: GPU-accelerated audio preprocessing
4. **Memory optimization**: Leverage AMD Infinity Cache characteristics

### 3.3 Training and Optimization

**From KAN paper (Section 7-10):**
- **Adaptive sampling/grids**: Consider adaptive grid refinement for KAN layers
- **Sparsity & regularization**: Implement L1/L2 regularization on KAN parameters
- **Scaling laws**: Monitor convergence rates vs. grid size, basis choice

**From KernelEvolve paper:**
- **Profiling**: Use unified profiling (Triton MPP equivalent) for kernel optimization
- **Evaluation framework**: Automated kernel correctness testing
- **Knowledge base**: Document hardware-specific constraints for AMD RX 7900 XTX

---

## 4. Recommendations

### 4.1 Immediate Actions

1. **Validate basis function choices** against paper's performance tables
2. **Profile GPU kernels** to identify bottlenecks
3. **Consider Triton migration** for improved portability and optimization

### 4.2 Future Enhancements

1. **Adaptive grid refinement**: Implement dynamic grid sizing based on training progress
2. **Kernel fusion**: Fuse adjacent KAN operations in the pipeline
3. **Multi-platform support**: Extend beyond AMD to NVIDIA via Triton
4. **Automated kernel generation**: Explore KernelEvolve-style agentic kernel coding for new operators

### 4.3 Documentation Updates

1. **Basis function selection rationale**: Document why each KAN type was chosen (already in `implementation_plan.md`)
2. **GPU optimization notes**: Document AMD-specific optimizations and constraints
3. **Performance benchmarks**: Compare against paper's reported performance trends

---

## 5. Key Takeaways

1. **Basis function choice is critical**: Our multi-basis architecture aligns with paper's guidance
2. **GPU kernel coverage matters**: All components need GPU implementations, not just compute-intensive ones
3. **Triton offers portability**: Consider Triton for future multi-platform support
4. **Performance vs. development time**: Balance optimization effort with development velocity
5. **Hardware-specific knowledge**: Document AMD RX 7900 XTX characteristics for optimization

---

## References

1. Noorizadegan, A., Wang, S., & Ling, L. (2025). A Practitioner's Guide to Kolmogorov–Arnold Networks. arXiv:2510.25781v2
2. KernelEvolve Team, Meta Platforms (2025). KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta. arXiv:2512.23236v2

---

*Last updated: Based on PDFs provided as context for the software implementation.*
