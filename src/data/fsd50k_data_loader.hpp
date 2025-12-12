#pragma once

#include "../audio/audio_buffer.hpp"
#include "../audio/preprocessing.hpp"
#include "../audio/wav_reader.hpp"
#include "../core/tensor.hpp"
#include "fsd50k_loader.hpp"
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <iostream>
#include <cmath>
#include <filesystem>

// Enhanced FSD50K dataset loader with audio loading
class FSD50KDataLoader {
public:
    FSD50KDataLoader(
        const std::string& dataset_path,
        const std::string& split = "dev",  // "dev", "train", "val", "test"
        size_t batch_size = 32
    ) : dataset_path_(dataset_path),
        split_(split),
        batch_size_(batch_size),
        loader_(dataset_path),
        rng_(42) {
        
        // Load clip information
        load_clip_list();
    }
    
    // Load list of clips for the split
    void load_clip_list() {
        // Try different possible paths for CSV files
        std::vector<std::string> possible_paths;
        
        if (split_ == "dev") {
            possible_paths = {
                dataset_path_ + "/FSD50K.metadata/collection/collection_dev.csv",
                dataset_path_ + "/metadata/collection/collection_dev.csv",
                dataset_path_ + "/labels/dev.csv",
                dataset_path_ + "/dev.csv"
            };
        } else if (split_ == "train") {
            possible_paths = {
                dataset_path_ + "/FSD50K.metadata/collection/collection_train.csv",
                dataset_path_ + "/metadata/collection/collection_train.csv",
                dataset_path_ + "/labels/train.csv",
                dataset_path_ + "/train.csv"
            };
        } else if (split_ == "val" || split_ == "eval") {
            possible_paths = {
                dataset_path_ + "/FSD50K.metadata/collection/collection_eval.csv",
                dataset_path_ + "/metadata/collection/collection_eval.csv",
                dataset_path_ + "/labels/val.csv",
                dataset_path_ + "/val.csv"
            };
        } else {
            throw std::runtime_error("Unknown split: " + split_);
        }
        
        std::ifstream file;
        bool found = false;
        std::string csv_path;
        for (const auto& path : possible_paths) {
            file.open(path);
            if (file.is_open()) {
                csv_path = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
            throw std::runtime_error("Cannot open CSV file for split: " + split_);
        }
        
        file.close();
        
        clips_ = loader_.load_clips(csv_path);
        
        std::cout << "Loaded " << clips_.size() << " clips from: " << csv_path << std::endl;
        
        // Filter by split if needed (for dev set)
        // Note: For dev set, we need to filter by split field
        // For now, keep all clips (can be filtered later if needed)
        
        std::cout << "Loaded " << clips_.size() << " clips for split: " << split_ << std::endl;
    }
    
    // Get a batch of data
    struct DataBatch {
        std::vector<AudioBuffer> audio;
        std::vector<Tensor> labels;
        std::vector<std::string> clip_names;
    };
    
    DataBatch get_batch(size_t batch_idx) {
        DataBatch batch;
        
        size_t start_idx = (batch_idx * batch_size_) % clips_.size();
        
        for (size_t i = 0; i < batch_size_ && (start_idx + i) < clips_.size(); ++i) {
            size_t clip_idx = start_idx + i;
            const auto& clip = clips_[clip_idx];
            
            // Try to load audio (fallback to synthetic if file not found)
            AudioBuffer audio = load_audio_safe(clip.fname);
            
            // Encode labels
            Tensor labels = loader_.encode_labels(clip.labels);
            
            batch.audio.push_back(audio);
            batch.labels.push_back(labels);
            batch.clip_names.push_back(clip.fname);
        }
        
        return batch;
    }
    
    // Get number of batches
    size_t num_batches() const {
        return (clips_.size() + batch_size_ - 1) / batch_size_;
    }
    
    // Get total number of clips
    size_t num_clips() const {
        return clips_.size();
    }
    
    // Shuffle dataset
    void shuffle() {
        std::shuffle(clips_.begin(), clips_.end(), rng_);
    }
    
    // Get vocabulary
    const std::map<std::string, std::string>& vocabulary() const {
        return loader_.vocabulary();
    }
    
    size_t num_classes() const {
        return loader_.num_classes();
    }
    
private:
    std::string dataset_path_;
    std::string split_;
    size_t batch_size_;
    FSD50KLoader loader_;
    std::vector<FSD50KLoader::ClipInfo> clips_;
    std::mt19937 rng_;
    
    // Load audio safely (with fallback)
    AudioBuffer load_audio_safe(const std::string& fname) {
        // Try different possible audio paths
        std::vector<std::string> possible_paths;
        
        if (split_ == "dev") {
            possible_paths = {
                dataset_path_ + "/FSD50K.dev_audio/" + fname + ".wav",
                dataset_path_ + "/FSD50K.dev_audio_16k/" + fname + ".wav",
                dataset_path_ + "/dev_audio/" + fname + ".wav",
                dataset_path_ + "/clips/dev/" + fname + ".wav",
                dataset_path_ + "/clips/" + split_ + "/" + fname + ".wav"
            };
        } else if (split_ == "eval" || split_ == "val") {
            possible_paths = {
                dataset_path_ + "/FSD50K.eval_audio/" + fname + ".wav",
                dataset_path_ + "/FSD50K.eval_audio_16k/" + fname + ".wav",
                dataset_path_ + "/eval_audio/" + fname + ".wav",
                dataset_path_ + "/clips/eval/" + fname + ".wav"
            };
        } else {
            possible_paths = {
                dataset_path_ + "/clips/" + split_ + "/" + fname + ".wav",
                dataset_path_ + "/" + split_ + "/" + fname + ".wav"
            };
        }
        
        AudioBuffer audio;
        for (const auto& path : possible_paths) {
            if (WAVReader::read(path, audio)) {
                // Successfully loaded
                return audio;
            }
        }
        
        // Fallback: generate synthetic audio
        return generate_synthetic_audio(fname);
    }
    
    // Generate synthetic audio for testing
    AudioBuffer generate_synthetic_audio(const std::string& fname) {
        // Use filename hash to generate consistent synthetic audio
        size_t hash = std::hash<std::string>{}(fname);
        double frequency = 440.0 + (hash % 1000) * 0.1;
        size_t duration_samples = 22050 + (hash % 11025);  // 0.5-1.0 seconds
        
        AudioBuffer audio(duration_samples, 44100, 1);
        for (size_t j = 0; j < audio.num_samples(); ++j) {
            double t = static_cast<double>(j) / audio.sample_rate();
            audio[j] = static_cast<float>(0.3 * std::sin(2.0 * M_PI * frequency * t));
        }
        
        // Normalize
        AudioPreprocessor::normalize(audio);
        
        return audio;
    }
};

