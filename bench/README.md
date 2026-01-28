# KAN Benchmark Suite

This directory contains synthetic benchmark tasks intended to stress Kolmogorov-Arnold Networks (KAN) in regimes where spline interpolation struggles and MLP-style baselines regain ground.

## Build

Enable the `bench` executable via the standard CMake build:

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/bench --task discontinuous_step --model kan --seed 123 --epochs 200 --report out.json
```

Common tasks:
- `discontinuous_step`, `discontinuous_sign`, `discontinuous_piecewise`
- `noisy_regression_gaussian`, `noisy_regression_student`, `noisy_moons`
- `adversarial_moons`, `adversarial_regression`
- `categorical_bag`, `categorical_sequence`

Model choices:
- `kan` (KAN spline layer)
- `mlp` (MLP baseline)

For categorical tasks, `mlp` maps to an embedding + pooling + MLP head, while `kan` maps to embedding + pooling + KAN head. A Transformer-lite baseline is not included yet; the embedding baseline is used instead.

## Reports

Reports are emitted as JSON. Example fields:
- `train`, `val`, `test` metrics (MSE/MAE or accuracy/cross-entropy)
- `robustness` arrays for perturbation sweeps
- `parameters` and rough `flops`

Discontinuous KAN runs default to a grid-size sweep (`16, 32, 64, 128`) and emit an array of JSON objects.
