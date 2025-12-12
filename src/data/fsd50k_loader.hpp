#pragma once

#include "../audio/audio_buffer.hpp"
#include "../core/tensor.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

// FSD50K dataset loader
// Format: PCM 16-bit, 44.1 kHz mono WAV files
class FSD50KLoader {
public:
    struct Label {
        std::string class_name;
        std::string mid;  // Freebase identifier
    };
    
    struct ClipInfo {
        std::string fname;  // File name without .wav extension
        std::vector<Label> labels;
        std::string split;  // "train" or "val" (only for dev set)
    };
    
    FSD50KLoader(const std::string& dataset_path) 
        : dataset_path_(dataset_path) {
        load_vocabulary();
    }
    
    // Load vocabulary (200 classes)
    void load_vocabulary() {
        std::string vocab_path = dataset_path_ + "/labels/vocabulary.csv";
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vocabulary.csv");
        }
        
        std::string line;
        std::getline(file, line);  // Skip header
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string class_name, mid;
            if (std::getline(iss, class_name, ',') && std::getline(iss, mid)) {
                vocabulary_[class_name] = mid;
                class_to_idx_[class_name] = class_to_idx_.size();
            }
        }
    }
    
    // Load clip info from CSV
    std::vector<ClipInfo> load_clips(const std::string& csv_path) {
        std::vector<ClipInfo> clips;
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + csv_path);
        }
        
        std::string line;
        std::getline(file, line);  // Skip header
        
        while (std::getline(file, line)) {
            ClipInfo clip;
            std::istringstream iss(line);
            std::string token;
            
            // Parse fname
            if (!std::getline(iss, clip.fname, ',')) continue;
            
            // Parse labels (semicolon-separated)
            std::string labels_str;
            if (std::getline(iss, labels_str, ',')) {
                std::istringstream labels_stream(labels_str);
                std::string label;
                while (std::getline(labels_stream, label, ';')) {
                    Label l;
                    l.class_name = label;
                    if (vocabulary_.count(label)) {
                        l.mid = vocabulary_[label];
                        clip.labels.push_back(l);
                    }
                }
            }
            
            // Parse mids (if present)
            std::string mids_str;
            if (std::getline(iss, mids_str, ',')) {
                // MIDs are also semicolon-separated
            }
            
            // Parse split (only for dev set)
            if (std::getline(iss, clip.split, ',')) {
                // Split is "train" or "val"
            }
            
            clips.push_back(clip);
        }
        
        return clips;
    }
    
    // Load audio file (placeholder - requires WAV reading implementation)
    AudioBuffer load_audio(const std::string& fname, const std::string& split = "dev") {
        std::string audio_path = dataset_path_ + "/clips/" + split + "/" + fname + ".wav";
        
        // For now, return empty buffer
        // In production, use libsndfile or implement WAV reader
        AudioBuffer buffer;
        
        // TODO: Implement actual WAV file reading
        // For testing, we can create synthetic audio
        
        return buffer;
    }
    
    // Encode labels to multi-hot vector (200 classes)
    Tensor encode_labels(const std::vector<Label>& labels) {
        Tensor encoded({static_cast<size_t>(vocabulary_.size())});
        encoded.fill(0.0);
        
        for (const auto& label : labels) {
            if (class_to_idx_.count(label.class_name)) {
                size_t idx = class_to_idx_[label.class_name];
                encoded[idx] = 1.0;
            }
        }
        
        return encoded;
    }
    
    // Get vocabulary
    const std::map<std::string, std::string>& vocabulary() const { return vocabulary_; }
    size_t num_classes() const { return vocabulary_.size(); }
    
    // Get class index
    int get_class_index(const std::string& class_name) const {
        if (class_to_idx_.count(class_name)) {
            return class_to_idx_.at(class_name);
        }
        return -1;
    }
    
private:
    std::string dataset_path_;
    std::map<std::string, std::string> vocabulary_;  // class_name -> mid
    std::map<std::string, int> class_to_idx_;        // class_name -> index
};

// Batch structure for training
struct AudioBatch {
    std::vector<Tensor> audio_features;  // Mel-spectrograms
    std::vector<Tensor> labels;          // Multi-hot encoded labels
    std::vector<std::string> clip_names; // For debugging
};

