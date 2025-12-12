#pragma once

#include "../core/tensor.hpp"
#include "../quantum/wavefunction.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>

// Training metrics
struct TrainingMetrics {
    double loss = 0.0;
    double audio_loss = 0.0;
    double quantum_loss = 0.0;
    double classification_loss = 0.0;
    double l2_loss = 0.0;
    double normalization_loss = 0.0;
    size_t num_samples = 0;
    size_t num_batches = 0;
};

// Model checkpoint
struct Checkpoint {
    size_t epoch = 0;
    size_t step = 0;
    double best_loss = 1e10;
    std::string checkpoint_path;
    std::chrono::system_clock::time_point timestamp;
    
    // Model state (simplified - in production, save actual parameters)
    std::vector<std::vector<double>> parameters;
};

// Forward declaration
class Trainer {
public:
    Trainer(
        size_t batch_size = 32,
        double learning_rate = 1e-4,
        size_t gradient_accumulation_steps = 4,
        const std::string& checkpoint_dir = "checkpoints"
    );
    
    // Training step
    TrainingMetrics train_step(
        const std::vector<Tensor>& audio_features,
        const std::vector<Tensor>& labels,
        const std::vector<Wavefunction>& quantum_embeddings
    );
    
    // Validation
    TrainingMetrics validate(
        const std::vector<Tensor>& audio_features,
        const std::vector<Tensor>& labels,
        const std::vector<Wavefunction>& quantum_embeddings
    );
    
    // Save checkpoint
    void save_checkpoint(const std::string& name, const Checkpoint& checkpoint);
    
    // Load checkpoint
    Checkpoint load_checkpoint(const std::string& name);
    
    // Get current training state
    size_t current_step() const;
    size_t current_epoch() const;
    double current_learning_rate() const;
    
    // Reset training state
    void reset();
    
private:
    size_t batch_size_;
    size_t gradient_accumulation_steps_;
    std::string checkpoint_dir_;
    
    AdamW* optimizer_;
    CosineAnnealingScheduler* scheduler_;
    MultiTaskLoss* loss_fn_;
    
    size_t current_step_;
    size_t current_epoch_;
};
