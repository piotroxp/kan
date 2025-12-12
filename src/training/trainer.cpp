#include "trainer.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Trainer::Trainer(
    size_t batch_size,
    double learning_rate,
    size_t gradient_accumulation_steps,
    const std::string& checkpoint_dir
) : batch_size_(batch_size),
    gradient_accumulation_steps_(gradient_accumulation_steps),
    checkpoint_dir_(checkpoint_dir),
    optimizer_(new AdamW(learning_rate)),
    scheduler_(new CosineAnnealingScheduler(learning_rate, learning_rate * 0.01, 100000)),
    loss_fn_(new MultiTaskLoss(1.0, 1.0, 1.0, 1e-5, 0.1)),
    current_step_(0),
    current_epoch_(0) {
}

TrainingMetrics Trainer::train_step(
    const std::vector<Tensor>& audio_features,
    const std::vector<Tensor>& labels,
    const std::vector<Wavefunction>& quantum_embeddings
) {
    TrainingMetrics metrics;
    
    // Simplified training step
    Tensor dummy_pred = audio_features[0];
    Tensor dummy_target = audio_features[0];
    
    std::vector<std::pair<size_t, size_t>> positive_pairs;
    std::vector<std::pair<size_t, size_t>> negative_pairs;
    
    for (size_t i = 0; i < audio_features.size() && i < 2; ++i) {
        positive_pairs.push_back({i, i});
    }
    
    auto loss_components = loss_fn_->compute(
        dummy_pred, dummy_target,
        quantum_embeddings,
        positive_pairs, negative_pairs,
        labels[0], labels[0],
        {}
    );
    
    metrics.loss = loss_components.total;
    metrics.audio_loss = loss_components.audio;
    metrics.quantum_loss = loss_components.quantum;
    metrics.classification_loss = loss_components.classification;
    metrics.l2_loss = loss_components.l2;
    metrics.normalization_loss = loss_components.normalization;
    metrics.num_samples = audio_features.size();
    metrics.num_batches = 1;
    
    current_step_++;
    
    double new_lr = scheduler_->get_lr(current_step_);
    optimizer_->set_learning_rate(new_lr);
    
    return metrics;
}

TrainingMetrics Trainer::validate(
    const std::vector<Tensor>& audio_features,
    const std::vector<Tensor>& labels,
    const std::vector<Wavefunction>& quantum_embeddings
) {
    return train_step(audio_features, labels, quantum_embeddings);
}

void Trainer::save_checkpoint(const std::string& name, const Checkpoint& checkpoint) {
    std::string path = checkpoint_dir_ + "/" + name + ".ckpt";
    std::ofstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open checkpoint file: " + path);
    }
    
    file << "epoch=" << checkpoint.epoch << "\n";
    file << "step=" << checkpoint.step << "\n";
    file << "best_loss=" << checkpoint.best_loss << "\n";
}

Checkpoint Trainer::load_checkpoint(const std::string& name) {
    std::string path = checkpoint_dir_ + "/" + name + ".ckpt";
    std::ifstream file(path);
    
    Checkpoint checkpoint;
    checkpoint.checkpoint_path = path;
    
    if (!file.is_open()) {
        return checkpoint;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            if (key == "epoch") {
                checkpoint.epoch = std::stoull(value);
            } else if (key == "step") {
                checkpoint.step = std::stoull(value);
            } else if (key == "best_loss") {
                checkpoint.best_loss = std::stod(value);
            }
        }
    }
    
    current_epoch_ = checkpoint.epoch;
    current_step_ = checkpoint.step;
    
    return checkpoint;
}

size_t Trainer::current_step() const {
    return current_step_;
}

size_t Trainer::current_epoch() const {
    return current_epoch_;
}

double Trainer::current_learning_rate() const {
    return optimizer_->learning_rate();
}

void Trainer::reset() {
    current_step_ = 0;
    current_epoch_ = 0;
    optimizer_->reset();
}
