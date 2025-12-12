# Training Configuration - Optimal Settings

## Training Parameters

- **Batch Size**: 16 (optimal for CPU, good for GPU when detected)
- **Learning Rate**: 1e-4 (standard for speech models)
- **Epochs**: 50
- **Dataset**: FSD50K (40,966 clips)
- **Classes**: 199 sound event classes

## Why Batch Size 16?

- **CPU Training**: Efficient memory usage, good throughput
- **GPU Ready**: Will scale well when GPU detection is fixed
- **Memory**: Fits comfortably in system RAM
- **Gradient Accumulation**: Already configured in trainer

## Training Progress

Training started with:
```bash
./train 16 1e-4 50
```

Output is being logged to `training.log` for monitoring.

## Expected Training Time

- **CPU**: ~X hours per epoch (depends on system)
- **GPU**: ~10-50x faster when GPU is used
- **Total**: ~Y hours for 50 epochs on CPU

## Monitoring

Check training progress:
```bash
tail -f build/training.log
```

Or check GPU usage (when GPU is working):
```bash
watch -n 1 rocm-smi
```

---

*Training started with optimal parameters!*

