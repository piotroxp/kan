#pragma once

#include "../model/speech_model.hpp"
#include "../training/trainer.hpp"
#include "../training/loss.hpp"
#include "../audio/audio_buffer.hpp"
#include "../audio/preprocessing.hpp"
#include "../core/tensor.hpp"
#include "../quantum/wavefunction.hpp"
#include "../gpu/gpu_kan_layer.hpp"
#include "../gpu/rocm_manager.hpp"
#include "../training/gpu_config.hpp"
#include "../training/backprop.hpp"
#include "../training/evaluation.hpp"
#include "../training/checkpoint_manager.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>

// Training batch
struct TrainingBatch {
    std::vector<AudioBuffer> audio;
    std::vector<Tensor> labels;  // Multi-hot encoded (200 classes)
    std::vector<std::string> clip_names;
};

// Batch generator for training
class BatchGenerator {
public:
    BatchGenerator(size_t batch_size) : batch_size_(batch_size), rng_(42) {}
    
    // Generate synthetic batch for training
    TrainingBatch generate_synthetic_batch() {
        TrainingBatch batch;
        
        for (size_t i = 0; i < batch_size_; ++i) {
            // Create synthetic audio (different frequencies)
            double frequency = 440.0 + (i % 10) * 100.0;
            size_t duration_samples = 22050 + (i % 5) * 11025;  // 0.5-1.0 seconds
            
            AudioBuffer audio(duration_samples, 44100, 1);
            for (size_t j = 0; j < audio.num_samples(); ++j) {
                double t = static_cast<double>(j) / audio.sample_rate();
                audio[j] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * frequency * t));
            }
            
            // Normalize
            AudioPreprocessor::normalize(audio);
            
            batch.audio.push_back(audio);
            
            // Create random multi-label (1-3 classes per sample)
            Tensor label({200});
            label.fill(0.0);
            int num_labels = 1 + (i % 3);
            std::uniform_int_distribution<int> class_dist(0, 199);
            for (int k = 0; k < num_labels; ++k) {
                int class_idx = class_dist(rng_);
                label[class_idx] = 1.0;
            }
            
            batch.labels.push_back(label);
            batch.clip_names.push_back("synthetic_" + std::to_string(i));
        }
        
        return batch;
    }
    
private:
    size_t batch_size_;
    std::mt19937 rng_;
};

// Training session
class TrainingSession {
public:
    TrainingSession(
        const std::string& checkpoint_dir = "checkpoints",
        size_t batch_size = 32,
        double learning_rate = 1e-4,
        size_t num_epochs = 10
    ) : checkpoint_dir_(checkpoint_dir),
        model_(80, 256, 200, 1024),
        trainer_(batch_size, learning_rate, 4, checkpoint_dir),
        batch_generator_(batch_size),
        num_epochs_(num_epochs),
        batch_size_(batch_size),
        gpu_manager_(),
        use_gpu_(gpu_manager_.is_gpu_available()) {
        
        // Use larger batch size for GPU if available
        if (use_gpu_ && batch_size < 64) {
            std::cout << "GPU detected: Using batch size " << batch_size 
                      << " (consider increasing to 64+ for better GPU utilization)" << std::endl;
        }
        
        // Create checkpoint directory
        std::string mkdir_cmd = "mkdir -p " + checkpoint_dir_;
        system(mkdir_cmd.c_str());
        
        // Try to load checkpoint if exists
        if (CheckpointManager::checkpoint_exists(checkpoint_dir + "/best_model.ckpt")) {
            std::cout << "Loading checkpoint from: " << checkpoint_dir << "/best_model.ckpt" << std::endl;
            Checkpoint loaded = CheckpointManager::load_model_checkpoint(
                checkpoint_dir + "/best_model.ckpt",
                model_
            );
            if (loaded.epoch > 0) {
                std::cout << "  Resumed from epoch: " << loaded.epoch << ", step: " << loaded.step << std::endl;
                std::cout << "  Best loss: " << loaded.best_loss << std::endl;
            }
        }
        
        if (use_gpu_) {
            std::cout << "Training will use GPU acceleration" << std::endl;
        } else {
            std::cout << "Training will use CPU (GPU not available)" << std::endl;
        }
    }
    
    // Train for one epoch
    TrainingMetrics train_epoch(size_t epoch, size_t num_batches = 100) {
        TrainingMetrics epoch_metrics;
        epoch_metrics.num_batches = 0;
        epoch_metrics.num_samples = 0;
        
        double total_loss = 0.0;
        double total_audio_loss = 0.0;
        double total_quantum_loss = 0.0;
        double total_classification_loss = 0.0;
        double total_l2_loss = 0.0;
        double total_normalization_loss = 0.0;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Generate batch
            TrainingBatch batch = batch_generator_.generate_synthetic_batch();
            
            // Process entire batch on GPU at once (CRITICAL for GPU utilization)
            if (batch_idx % 10 == 0) {
                std::cout << "  Processing batch " << (batch_idx + 1) << " (size: " << batch.audio.size() << ")..." << std::endl;
            }
            std::vector<SpeechModel::ModelOutput> outputs = model_.forward_batch(batch.audio);
            if (batch_idx % 10 == 0) {
                std::cout << "  Batch processing complete" << std::endl;
            }
            
            // Extract features from batch outputs
            std::vector<Tensor> audio_features;
            std::vector<Wavefunction> quantum_embeddings;
            audio_features.reserve(outputs.size());
            quantum_embeddings.reserve(outputs.size());
            
            for (const auto& output : outputs) {
                audio_features.push_back(output.audio_features);
                quantum_embeddings.push_back(output.quantum_embeddings[0]);
            }
            
            // Create positive/negative pairs for quantum loss
            std::vector<std::pair<size_t, size_t>> positive_pairs;
            std::vector<std::pair<size_t, size_t>> negative_pairs;
            
            // Positive pairs: same batch items with similar labels
            for (size_t i = 0; i < batch.labels.size(); ++i) {
                for (size_t j = i + 1; j < batch.labels.size(); ++j) {
                    // Check label similarity (simplified)
                    double similarity = compute_label_similarity(batch.labels[i], batch.labels[j]);
                    if (similarity > 0.5) {
                        positive_pairs.push_back({i, j});
                    } else {
                        negative_pairs.push_back({i, j});
                    }
                }
            }
            
            // Compute loss
            MultiTaskLoss loss_fn(1.0, 1.0, 1.0, 1e-5, 0.1);
            
            // Compute quantum loss for the batch
            double batch_quantum_loss = 0.0;
            if (!positive_pairs.empty() || !negative_pairs.empty()) {
                batch_quantum_loss = loss_fn.quantum_fidelity_loss(
                    quantum_embeddings, positive_pairs, negative_pairs);
            }
            
            // For each sample in batch
            double batch_loss = 0.0;
            double batch_audio_loss = 0.0;
            double batch_classification_loss = 0.0;
            double batch_l2_loss = 0.0;
            double batch_normalization_loss = 0.0;
            
            for (size_t i = 0; i < batch.audio.size(); ++i) {
                auto output = model_.forward(batch.audio[i]);
                
                // Dummy audio reconstruction (in production, add decoder)
                Tensor audio_pred = output.audio_features;
                Tensor audio_target = output.audio_features;  // Simplified
                
                auto loss_components = loss_fn.compute(
                    audio_pred, audio_target,
                    {output.quantum_embeddings[0]},
                    {}, {},  // Quantum pairs computed above separately
                    output.classification_logits,
                    batch.labels[i],
                    model_.get_parameters()
                );
                
                batch_loss += loss_components.total;
                batch_audio_loss += loss_components.audio;
                batch_classification_loss += loss_components.classification;
                batch_l2_loss += loss_components.l2;
                batch_normalization_loss += loss_components.normalization;
            }
            
            // Average over batch
            size_t batch_size_actual = batch.audio.size();
            batch_loss /= batch_size_actual;
            batch_audio_loss /= batch_size_actual;
            batch_classification_loss /= batch_size_actual;
            batch_l2_loss /= batch_size_actual;
            batch_normalization_loss /= batch_size_actual;
            
            // Add quantum loss (already averaged in loss function)
            batch_loss += batch_quantum_loss;
            
            // Compute gradients using backpropagation
            if (batch_size_actual > 0) {
                // Create loss gradient (simplified - in production compute from loss)
                Tensor loss_grad({static_cast<size_t>(model_.num_classes())});
                loss_grad.fill(batch_loss / model_.num_classes());
                
                // Get model output for first sample (for gradient computation)
                auto sample_output = model_.forward(batch.audio[0]);
                
                // Convert to ModelOutput format
                ModelOutput model_output;
                model_output.audio_features = sample_output.audio_features;
                model_output.quantum_embeddings = sample_output.quantum_embeddings;
                model_output.semantic_representation = sample_output.semantic_representation;
                model_output.classification_logits = sample_output.classification_logits;
                
                // Backward pass
                auto grads = ModelBackward::backward(model_output, loss_grad, model_);
                
                // Update parameters using gradients (simplified - in production use optimizer)
                update_parameters_with_gradients(grads);
            }
            
            // Accumulate metrics
            total_loss += batch_loss;
            total_audio_loss += batch_audio_loss;
            total_quantum_loss += batch_quantum_loss;
            total_classification_loss += batch_classification_loss;
            total_l2_loss += batch_l2_loss;
            total_normalization_loss += batch_normalization_loss;
            
            epoch_metrics.num_batches++;
            epoch_metrics.num_samples += batch_size_actual;
            
            // Log progress
            if ((batch_idx + 1) % 10 == 0) {
                std::cout << "  Batch " << (batch_idx + 1) << "/" << num_batches 
                          << " - Loss: " << batch_loss << std::endl;
            }
        }
        
        // Average metrics
        epoch_metrics.loss = total_loss / epoch_metrics.num_batches;
        epoch_metrics.audio_loss = total_audio_loss / epoch_metrics.num_batches;
        epoch_metrics.quantum_loss = total_quantum_loss / epoch_metrics.num_batches;
        epoch_metrics.classification_loss = total_classification_loss / epoch_metrics.num_batches;
        epoch_metrics.l2_loss = total_l2_loss / epoch_metrics.num_batches;
        epoch_metrics.normalization_loss = total_normalization_loss / epoch_metrics.num_batches;
        
        return epoch_metrics;
    }
    
    // Run full training
    void train() {
        std::cout << "=== Starting Training ===" << std::endl;
        std::cout << "Model: SpeechModel with quantum embeddings" << std::endl;
        std::cout << "Batch size: " << batch_size_ << std::endl;
        std::cout << "Learning rate: " << trainer_.current_learning_rate() << std::endl;
        std::cout << "Epochs: " << num_epochs_ << std::endl;
        std::cout << std::endl;
        
        double best_loss = 1e10;
        
        for (size_t epoch = 0; epoch < num_epochs_; ++epoch) {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs_ << std::endl;
            
            auto metrics = train_epoch(epoch, 50);  // 50 batches per epoch
            
        std::cout << "  Loss: " << metrics.loss << std::endl;
        std::cout << "    Audio: " << metrics.audio_loss << std::endl;
        std::cout << "    Quantum: " << metrics.quantum_loss << std::endl;
        std::cout << "    Classification: " << metrics.classification_loss << std::endl;
        std::cout << "    L2: " << metrics.l2_loss << std::endl;
        std::cout << "    Normalization: " << metrics.normalization_loss << std::endl;
        std::cout << "  Samples: " << metrics.num_samples << std::endl;
        
        // Compute evaluation metrics on validation set (if available)
        if (epoch % 5 == 0) {  // Every 5 epochs
            compute_validation_metrics(epoch);
        }
        
        std::cout << std::endl;
            
            // Save checkpoint if best
            if (metrics.loss < best_loss) {
                best_loss = metrics.loss;
                Checkpoint checkpoint;
                checkpoint.epoch = epoch + 1;
                checkpoint.step = trainer_.current_step();
                checkpoint.best_loss = best_loss;
                trainer_.save_checkpoint("best_model", checkpoint);
                std::cout << "  Saved checkpoint (loss: " << best_loss << ")" << std::endl;
            }
        }
        
        std::cout << "=== Training Complete ===" << std::endl;
    }
    
private:
    std::string checkpoint_dir_;
    SpeechModel model_;
    Trainer trainer_;
    BatchGenerator batch_generator_;
    size_t num_epochs_;
    size_t batch_size_;
    ROCmMemoryManager gpu_manager_;
    bool use_gpu_;
    
    // Compute validation metrics
    void compute_validation_metrics(size_t epoch) {
        // Generate a small validation batch
        TrainingBatch val_batch = batch_generator_.generate_synthetic_batch();
        
        std::vector<Tensor> predictions;
        std::vector<Tensor> labels;
        std::vector<Wavefunction> quantum_embeddings;
        
        for (size_t i = 0; i < val_batch.audio.size() && i < 10; ++i) {
            auto output = model_.forward(val_batch.audio[i]);
            predictions.push_back(output.classification_logits);
            labels.push_back(val_batch.labels[i]);
            quantum_embeddings.push_back(output.quantum_embeddings[0]);
        }
        
        // Compute metrics
        double map = EvaluationMetrics::mean_average_precision(predictions, labels);
        double macro_f1 = EvaluationMetrics::macro_f1_score(predictions, labels);
        double micro_f1 = EvaluationMetrics::micro_f1_score(predictions, labels);
        
        auto quantum_metrics = EvaluationMetrics::compute_quantum_metrics(quantum_embeddings);
        
        std::cout << "  Validation Metrics:" << std::endl;
        std::cout << "    mAP: " << map << std::endl;
        std::cout << "    Macro F1: " << macro_f1 << std::endl;
        std::cout << "    Micro F1: " << micro_f1 << std::endl;
        std::cout << "    Quantum Avg Fidelity: " << quantum_metrics.avg_fidelity << std::endl;
        std::cout << "    Quantum Normalization: " << quantum_metrics.avg_normalization << std::endl;
    }
    
    double compute_label_similarity(const Tensor& label1, const Tensor& label2) {
        double dot_product = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (size_t i = 0; i < label1.size(); ++i) {
            dot_product += label1[i] * label2[i];
            norm1 += label1[i] * label1[i];
            norm2 += label2[i] * label2[i];
        }
        
        if (norm1 < 1e-10 || norm2 < 1e-10) {
            return 0.0;
        }
        
        return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    
    // Update parameters using computed gradients
    void update_parameters_with_gradients(const ModelBackward::Gradients& grads) {
        // Get current parameters
        auto params = model_.get_parameters();
        
        // Update parameters using gradients (simplified - in production use optimizer)
        double learning_rate = 1e-4;
        
        for (size_t i = 0; i < params.size() && i < grads.param_grads.size(); ++i) {
            for (size_t j = 0; j < params[i].size() && j < grads.param_grads[i].size(); ++j) {
                params[i][j] -= learning_rate * grads.param_grads[i][j];
            }
        }
        
        // Set updated parameters (in production, use proper parameter setting)
        // model_.set_parameters(params);
    }
};

