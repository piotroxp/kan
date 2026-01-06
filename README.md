# KAN Speech Model

A speech recognition and generation model featuring quantum embeddings-based understanding core and Kolmogorov-Arnold Network (KAN) architecture.

## Quick Start

See [docs/BUILD_README.md](docs/BUILD_README.md) for building and usage instructions.

## Documentation

All documentation is located in the [`docs/`](docs/) directory:

- **[BUILD_README.md](docs/BUILD_README.md)** - Build instructions, project structure, and component overview
- **[RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md)** - Research papers context and design rationale
- **[implementation_plan.md](docs/implementation_plan.md)** - Detailed implementation plan
- **[COMPLETE_IMPLEMENTATION.md](docs/COMPLETE_IMPLEMENTATION.md)** - Implementation status and architecture

### Implementation Status

- **[COMPLETE_IMPLEMENTATION.md](docs/COMPLETE_IMPLEMENTATION.md)** - Complete implementation summary
- **[PHASE2_SUMMARY.md](docs/PHASE2_SUMMARY.md)** - Phase 2: Audio Processing
- **[PHASE3_SUMMARY.md](docs/PHASE3_SUMMARY.md)** - Phase 3: Quantum Embeddings
- **[PHASE4_SUMMARY.md](docs/PHASE4_SUMMARY.md)** - Phase 4: Training Pipeline
- **[PHASE5_SUMMARY.md](docs/PHASE5_SUMMARY.md)** - Phase 5: Full Model Integration

### GPU Integration

- **[GPU_SETUP.md](docs/GPU_SETUP.md)** - GPU setup and configuration
- **[GPU_KERNELS_COMPLETE.md](docs/GPU_KERNELS_COMPLETE.md)** - GPU kernel implementation status
- **[GPU_INTEGRATION_COMPLETE.md](docs/GPU_INTEGRATION_COMPLETE.md)** - GPU integration summary

### Training Status

- **[TRAINING_STATUS.md](docs/TRAINING_STATUS.md)** - Current training status
- **[TRAINING_ACTIVE.md](docs/TRAINING_ACTIVE.md)** - Active training information
- **[TRAINING_RUNNING.md](docs/TRAINING_RUNNING.md)** - Training execution details

## Features

- **6 KAN Variants**: B-spline, Chebyshev, Sinc, Fourier, RBF, Piecewise Linear
- **Quantum Field Embeddings**: Squeezed coherent states with Born-rule fidelity
- **Audio Processing**: Mel-spectrogram extraction, preprocessing, augmentation
- **Multi-Task Training**: Audio, quantum, classification losses
- **AMD GPU Support**: ROCm/HIP kernels for AMD Radeon RX 7900 XTX
- **Full Model Pipeline**: Audio → Quantum → Semantic → Classification

## Project Structure

```
kan/
├── docs/              # All documentation
├── src/               # Source code
│   ├── core/         # KAN layer implementations
│   ├── quantum/      # Quantum wavefunction and embeddings
│   ├── audio/        # Audio processing
│   ├── model/        # Speech model + Language model
│   ├── training/     # Loss, optimizer, trainer
│   ├── inference/    # Inference engine
│   └── gpu/          # GPU kernels (ROCm/HIP)
├── tests/            # Unit tests
├── tools/            # Training tools
└── build.sh          # Build script
```

## License

See LICENSE file for details.
