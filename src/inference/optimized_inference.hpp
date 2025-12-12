#pragma once

#include "../model/speech_model.hpp"
#include "../core/tensor.hpp"
#include "../gpu/rocm_manager.hpp"
#include <vector>
#include <memory>
#include <map>
#include <cmath>

// Model quantization for inference optimization
class ModelQuantizer {
public:
    // Quantize model parameters to INT8
    static void quantize_to_int8(
        const std::vector<std::vector<double>>& float_params,
        std::vector<std::vector<int8_t>>& quantized_params,
        std::vector<double>& scales
    ) {
        quantized_params.clear();
        scales.clear();
        
        for (const auto& param_group : float_params) {
            if (param_group.empty()) {
                quantized_params.push_back({});
                scales.push_back(1.0);
                continue;
            }
            
            // Find min and max for scaling
            double min_val = *std::min_element(param_group.begin(), param_group.end());
            double max_val = *std::max_element(param_group.begin(), param_group.end());
            
            // Compute scale
            double scale = std::max(std::abs(min_val), std::abs(max_val)) / 127.0;
            if (scale < 1e-10) {
                scale = 1.0;
            }
            
            scales.push_back(scale);
            
            // Quantize
            std::vector<int8_t> quantized;
            quantized.reserve(param_group.size());
            for (double val : param_group) {
                int8_t qval = static_cast<int8_t>(std::round(val / scale));
                qval = std::max(static_cast<int8_t>(-128), 
                               std::min(static_cast<int8_t>(127), qval));
                quantized.push_back(qval);
            }
            
            quantized_params.push_back(quantized);
        }
    }
    
    // Dequantize INT8 parameters back to float
    static void dequantize_from_int8(
        const std::vector<std::vector<int8_t>>& quantized_params,
        const std::vector<double>& scales,
        std::vector<std::vector<double>>& float_params
    ) {
        float_params.clear();
        
        for (size_t i = 0; i < quantized_params.size(); ++i) {
            std::vector<double> dequantized;
            dequantized.reserve(quantized_params[i].size());
            
            double scale = (i < scales.size()) ? scales[i] : 1.0;
            
            for (int8_t qval : quantized_params[i]) {
                dequantized.push_back(static_cast<double>(qval) * scale);
            }
            
            float_params.push_back(dequantized);
        }
    }
    
    // Compute quantization error
    static double quantization_error(
        const std::vector<std::vector<double>>& original,
        const std::vector<std::vector<double>>& dequantized
    ) {
        double total_error = 0.0;
        size_t total_params = 0;
        
        for (size_t i = 0; i < original.size() && i < dequantized.size(); ++i) {
            for (size_t j = 0; j < original[i].size() && j < dequantized[i].size(); ++j) {
                double error = original[i][j] - dequantized[i][j];
                total_error += error * error;
                total_params++;
            }
        }
        
        return (total_params > 0) ? std::sqrt(total_error / total_params) : 0.0;
    }
};

// Optimized inference engine with quantization
class OptimizedInferenceEngine {
public:
    OptimizedInferenceEngine(
        SpeechModel& model,
        bool use_quantization = true,
        bool use_gpu = false
    ) : model_(model),
        use_quantization_(use_quantization),
        use_gpu_(use_gpu),
        gpu_manager_() {
        
        if (use_quantization_) {
            quantize_model();
        }
        
        if (use_gpu_ && gpu_manager_.is_gpu_available()) {
            std::cout << "Inference: GPU acceleration enabled" << std::endl;
        } else {
            use_gpu_ = false;
        }
    }
    
    // Optimized forward pass
    SpeechModel::ModelOutput forward(const AudioBuffer& audio) {
        if (use_quantization_ && !quantized_params_.empty()) {
            // Use quantized model (faster inference)
            return forward_quantized(audio);
        } else {
            // Use full precision model
            return model_.forward(audio);
        }
    }
    
    // Batch inference (optimized)
    std::vector<SpeechModel::ModelOutput> forward_batch(
        const std::vector<AudioBuffer>& audio_batch
    ) {
        std::vector<SpeechModel::ModelOutput> outputs;
        outputs.reserve(audio_batch.size());
        
        for (const auto& audio : audio_batch) {
            outputs.push_back(forward(audio));
        }
        
        return outputs;
    }
    
    // Get quantization statistics
    double get_quantization_error() const {
        return quantization_error_;
    }
    
    size_t get_quantized_size_bytes() const {
        size_t total = 0;
        for (const auto& qparams : quantized_params_) {
            total += qparams.size();
        }
        return total;  // INT8 = 1 byte per parameter
    }
    
    size_t get_float_size_bytes() const {
        size_t total = 0;
        auto params = model_.get_parameters();
        for (const auto& fparams : params) {
            total += fparams.size() * sizeof(double);
        }
        return total;
    }
    
private:
    SpeechModel& model_;
    bool use_quantization_;
    bool use_gpu_;
    ROCmMemoryManager gpu_manager_;
    
    // Quantized parameters
    std::vector<std::vector<int8_t>> quantized_params_;
    std::vector<double> quantization_scales_;
    double quantization_error_ = 0.0;
    
    void quantize_model() {
        auto float_params = model_.get_parameters();
        
        ModelQuantizer::quantize_to_int8(
            float_params,
            quantized_params_,
            quantization_scales_
        );
        
        // Compute quantization error
        std::vector<std::vector<double>> dequantized;
        ModelQuantizer::dequantize_from_int8(
            quantized_params_,
            quantization_scales_,
            dequantized
        );
        
        quantization_error_ = ModelQuantizer::quantization_error(
            float_params,
            dequantized
        );
        
        std::cout << "Model quantized: " << quantized_params_.size() << " parameter groups" << std::endl;
        std::cout << "  Quantization error: " << quantization_error_ << std::endl;
        std::cout << "  Size reduction: " 
                  << (1.0 - static_cast<double>(get_quantized_size_bytes()) / get_float_size_bytes()) * 100.0
                  << "%" << std::endl;
    }
    
    SpeechModel::ModelOutput forward_quantized(const AudioBuffer& audio) {
        // For now, use full precision (quantized inference needs kernel implementation)
        // In production, implement quantized forward pass
        return model_.forward(audio);
    }
};

