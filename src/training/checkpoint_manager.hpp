#pragma once

#include "../training/trainer.hpp"
#include "../model/speech_model.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

// Enhanced checkpoint manager with model state saving/loading
class CheckpointManager {
public:
    // Save complete model checkpoint
    static void save_model_checkpoint(
        const std::string& path,
        SpeechModel& model,  // Non-const to call get_parameters()
        const Checkpoint& checkpoint
    ) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open checkpoint file: " + path);
        }
        
        // Write header
        file << "KAN_SPEECH_MODEL_CHECKPOINT_v1\n";
        file << "epoch=" << checkpoint.epoch << "\n";
        file << "step=" << checkpoint.step << "\n";
        file << "best_loss=" << checkpoint.best_loss << "\n";
        file << "timestamp=" << std::chrono::duration_cast<std::chrono::seconds>(
            checkpoint.timestamp.time_since_epoch()).count() << "\n";
        
        // Write model parameters
        auto params = model.get_parameters();
        file << "num_param_groups=" << params.size() << "\n";
        
        for (size_t i = 0; i < params.size(); ++i) {
            file << "group_" << i << "_size=" << params[i].size() << "\n";
            file << "group_" << i << "_data=";
            
            for (size_t j = 0; j < params[i].size(); ++j) {
                if (j > 0) file << ",";
                file << std::scientific << params[i][j];
            }
            file << "\n";
        }
        
        file.close();
    }
    
    // Load complete model checkpoint
    static Checkpoint load_model_checkpoint(
        const std::string& path,
        SpeechModel& model
    ) {
        Checkpoint checkpoint;
        checkpoint.checkpoint_path = path;
        
        std::ifstream file(path);
        if (!file.is_open()) {
            return checkpoint;  // Return empty checkpoint if file not found
        }
        
        std::string line;
        std::getline(file, line);  // Read header
        
        if (line != "KAN_SPEECH_MODEL_CHECKPOINT_v1") {
            throw std::runtime_error("Invalid checkpoint format");
        }
        
        // Read metadata
        while (std::getline(file, line) && line.find("num_param_groups") == std::string::npos) {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                if (key == "epoch") {
                    checkpoint.epoch = std::stoull(value);
                } else if (key == "step") {
                    checkpoint.step = std::stoull(value);
                } else if (key == "best_loss") {
                    checkpoint.best_loss = std::stod(value);
                } else if (key == "timestamp") {
                    checkpoint.timestamp = std::chrono::system_clock::time_point(
                        std::chrono::seconds(std::stoll(value)));
                }
            }
        }
        
        // Read parameter groups
        std::vector<std::vector<double>> params;
        
        if (line.find("num_param_groups") != std::string::npos) {
            size_t num_groups = std::stoull(line.substr(line.find('=') + 1));
            params.resize(num_groups);
            
            for (size_t i = 0; i < num_groups; ++i) {
                // Read size
                std::getline(file, line);
                size_t group_size = std::stoull(line.substr(line.find('=') + 1));
                
                // Read data
                std::getline(file, line);
                std::string data_str = line.substr(line.find('=') + 1);
                
                std::istringstream data_stream(data_str);
                std::string value;
                params[i].reserve(group_size);
                
                while (std::getline(data_stream, value, ',')) {
                    params[i].push_back(std::stod(value));
                }
            }
        }
        
        // Set model parameters
        model.set_parameters(params);
        
        return checkpoint;
    }
    
    // Check if checkpoint exists
    static bool checkpoint_exists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }
};

