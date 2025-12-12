#include <iostream>
#include "training/trainer.hpp"
#include "training/loss.hpp"
#include "training/optimizer.hpp"
#include "audio/audio_buffer.hpp"
#include "audio/feature_extraction.hpp"
#include "quantum/quantum_field_embedding.hpp"
#include "core/tensor.hpp"

int main() {
    std::cout << "=== Training Pipeline Example ===\n\n";
    
    // 1. Create trainer
    std::cout << "1. Initializing trainer...\n";
    Trainer trainer(32, 1e-4, 4, "checkpoints");
    std::cout << "   Batch size: 32\n";
    std::cout << "   Learning rate: " << trainer.current_learning_rate() << "\n";
    std::cout << "   Gradient accumulation: 4 steps\n\n";
    
    // 2. Create synthetic data
    std::cout << "2. Creating synthetic training data...\n";
    
    // Create audio features (mel-spectrograms)
    std::vector<Tensor> audio_features;
    std::vector<Tensor> labels;
    std::vector<Wavefunction> quantum_embeddings;
    
    SincKANMelSpectrogram extractor(80, 2048, 512, 2048, 0.0, 22050.0, 44100);
    QuantumFieldEmbeddingCore embedding(80, 1, 1024, 12.0, 1.5);
    
    for (int i = 0; i < 4; ++i) {
        // Create synthetic audio
        AudioBuffer audio(22050, 44100, 1);  // 0.5 seconds
        double frequency = 440.0 + i * 100.0;
        for (size_t j = 0; j < audio.num_samples(); ++j) {
            double t = static_cast<double>(j) / audio.sample_rate();
            audio[j] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * frequency * t));
        }
        
        // Extract features
        Tensor mel_spec = extractor.process(audio);
        
        // Pool over time (mean)
        Tensor pooled({static_cast<size_t>(mel_spec.shape()[1])});
        pooled.fill(0.0);
        for (size_t t = 0; t < mel_spec.shape()[0]; ++t) {
            for (size_t m = 0; m < mel_spec.shape()[1]; ++m) {
                pooled[m] += mel_spec.at({t, m});
            }
        }
        for (size_t m = 0; m < pooled.size(); ++m) {
            pooled[m] /= mel_spec.shape()[0];
        }
        
        audio_features.push_back(pooled);
        
        // Create labels (multi-hot, 200 classes)
        Tensor label({200});
        label.fill(0.0);
        label[i % 200] = 1.0;  // One class per sample
        labels.push_back(label);
        
        // Create quantum embedding
        auto wfs = embedding.encode(pooled);
        quantum_embeddings.push_back(wfs[0]);
    }
    
    std::cout << "   Created " << audio_features.size() << " samples\n";
    std::cout << "   Feature shape: [" << audio_features[0].shape()[0] << "]\n";
    std::cout << "   Label shape: [" << labels[0].shape()[0] << "]\n\n";
    
    // 3. Training step
    std::cout << "3. Running training step...\n";
    auto metrics = trainer.train_step(audio_features, labels, quantum_embeddings);
    
    std::cout << "   Total loss: " << metrics.loss << "\n";
    std::cout << "   Audio loss: " << metrics.audio_loss << "\n";
    std::cout << "   Quantum loss: " << metrics.quantum_loss << "\n";
    std::cout << "   Classification loss: " << metrics.classification_loss << "\n";
    std::cout << "   L2 loss: " << metrics.l2_loss << "\n";
    std::cout << "   Normalization loss: " << metrics.normalization_loss << "\n";
    std::cout << "   Step: " << trainer.current_step() << "\n";
    std::cout << "   Learning rate: " << trainer.current_learning_rate() << "\n\n";
    
    // 4. Validation
    std::cout << "4. Running validation...\n";
    auto val_metrics = trainer.validate(audio_features, labels, quantum_embeddings);
    std::cout << "   Validation loss: " << val_metrics.loss << "\n\n";
    
    // 5. Checkpointing
    std::cout << "5. Testing checkpointing...\n";
    Checkpoint checkpoint;
    checkpoint.epoch = 1;
    checkpoint.step = trainer.current_step();
    checkpoint.best_loss = metrics.loss;
    
    trainer.save_checkpoint("test_checkpoint", checkpoint);
    std::cout << "   Checkpoint saved\n";
    
    Checkpoint loaded = trainer.load_checkpoint("test_checkpoint");
    std::cout << "   Checkpoint loaded: epoch=" << loaded.epoch 
              << ", step=" << loaded.step << "\n\n";
    
    std::cout << "=== Training pipeline completed successfully! ===\n";
    
    return 0;
}


