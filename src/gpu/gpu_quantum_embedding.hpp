#pragma once

#include "../quantum/quantum_field_embedding.hpp"
#include "../quantum/wavefunction.hpp"
#include "../core/tensor.hpp"
#include "rocm_manager.hpp"
#include "quantum_kernels.hpp"
#include "gpu_kan_layer.hpp"
#include <vector>
#include <memory>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

// GPU-accelerated quantum embedding
class GPUQuantumEmbedding {
public:
    GPUQuantumEmbedding(
        int input_dim,
        int embedding_dim,
        int grid_size = 1024,
        double L = 12.0,
        double sigma = 1.5
    ) : input_dim_(input_dim),
        embedding_dim_(embedding_dim),
        grid_size_(grid_size),
        L_(L),
        sigma_(sigma),
        manager_(),
        use_gpu_(manager_.is_gpu_available()) {
        
        if (use_gpu_) {
            // Create GPU Chebyshev KAN layers for alpha, beta, gamma extraction
            gpu_alpha_extractor_ = std::make_unique<GPUKANLayer>(
                KANBasis::Chebyshev, input_dim, 1, 10, 8
            );
            gpu_beta_extractor_ = std::make_unique<GPUKANLayer>(
                KANBasis::Chebyshev, input_dim, 1, 10, 8
            );
            gpu_gamma_extractor_ = std::make_unique<GPUKANLayer>(
                KANBasis::Chebyshev, input_dim, 1, 10, 8
            );
            
            // Allocate GPU memory for wavefunctions
            size_t wavefunction_size = grid_size_ * sizeof(float) * 2;  // real + imag
            gpu_wavefunction_real_ = manager_.allocate(wavefunction_size);
            gpu_wavefunction_imag_ = manager_.allocate(wavefunction_size);
        } else {
            // CPU fallback
            cpu_embedding_ = std::make_unique<QuantumFieldEmbeddingCore>(
                input_dim, embedding_dim, grid_size, L, sigma
            );
        }
    }
    
    ~GPUQuantumEmbedding() {
        if (use_gpu_) {
            if (gpu_wavefunction_real_) manager_.free(gpu_wavefunction_real_);
            if (gpu_wavefunction_imag_) manager_.free(gpu_wavefunction_imag_);
        }
    }
    
    // Encode single feature vector
    std::vector<Wavefunction> encode(const Tensor& features) {
        if (use_gpu_) {
            return encode_gpu(features);
        } else {
            return cpu_embedding_->encode(features);
        }
    }
    
    // Encode batch on GPU
    std::vector<std::vector<Wavefunction>> encode_batch(const std::vector<Tensor>& features_batch) {
        if (use_gpu_) {
            return encode_batch_gpu(features_batch);
        } else {
            return cpu_embedding_->encode_batch(features_batch);
        }
    }
    
    bool is_using_gpu() const { return use_gpu_; }
    
    // Get parameters (for training)
    std::vector<double> get_alpha_params() const {
        if (use_gpu_ && gpu_alpha_extractor_) {
            return gpu_alpha_extractor_->parameters();
        }
        return cpu_embedding_->get_alpha_params();
    }
    
    std::vector<double> get_beta_params() const {
        if (use_gpu_ && gpu_beta_extractor_) {
            return gpu_beta_extractor_->parameters();
        }
        return cpu_embedding_->get_beta_params();
    }
    
    std::vector<double> get_gamma_params() const {
        if (use_gpu_ && gpu_gamma_extractor_) {
            return gpu_gamma_extractor_->parameters();
        }
        return cpu_embedding_->get_gamma_params();
    }
    
private:
    int input_dim_, embedding_dim_, grid_size_;
    double L_, sigma_;
    
    ROCmMemoryManager manager_;
    bool use_gpu_;
    
    // GPU layers
    std::unique_ptr<GPUKANLayer> gpu_alpha_extractor_;
    std::unique_ptr<GPUKANLayer> gpu_beta_extractor_;
    std::unique_ptr<GPUKANLayer> gpu_gamma_extractor_;
    
    // GPU memory for wavefunctions
    void* gpu_wavefunction_real_ = nullptr;
    void* gpu_wavefunction_imag_ = nullptr;
    
    // CPU fallback
    std::unique_ptr<QuantumFieldEmbeddingCore> cpu_embedding_;
    
    std::vector<Wavefunction> encode_gpu(const Tensor& features) {
#ifdef USE_HIP
        // Flatten features if needed
        Tensor flat_features = features;
        if (features.ndim() > 1) {
            size_t total_size = features.size();
            flat_features = features.reshape({total_size});
        }
        
        // Extract alpha, beta, gamma using GPU KAN layers
        Tensor alpha_tensor = gpu_alpha_extractor_->forward(flat_features);
        Tensor beta_tensor = gpu_beta_extractor_->forward(flat_features);
        Tensor gamma_tensor = gpu_gamma_extractor_->forward(flat_features);
        
        double alpha = alpha_tensor[0];
        double beta = beta_tensor[0];
        double gamma = gamma_tensor[0];
        
        // Compute wavefunction on GPU
        std::vector<float> alpha_vec = {static_cast<float>(alpha)};
        std::vector<float> beta_vec = {static_cast<float>(beta)};
        std::vector<float> gamma_vec = {static_cast<float>(gamma)};
        
        void* gpu_alpha = manager_.allocate(sizeof(float));
        void* gpu_beta = manager_.allocate(sizeof(float));
        void* gpu_gamma = manager_.allocate(sizeof(float));
        
        manager_.copy_to_device(alpha_vec.data(), gpu_alpha, sizeof(float));
        manager_.copy_to_device(beta_vec.data(), gpu_beta, sizeof(float));
        manager_.copy_to_device(gamma_vec.data(), gpu_gamma, sizeof(float));
        
        // Launch wavefunction computation kernel
        GPU::launch_compute_squeezed_coherent(
            static_cast<const float*>(gpu_alpha),
            static_cast<const float*>(gpu_beta),
            static_cast<const float*>(gpu_gamma),
            static_cast<float*>(gpu_wavefunction_real_),
            static_cast<float*>(gpu_wavefunction_imag_),
            1,  // batch_size = 1
            grid_size_,
            static_cast<float>(L_),
            static_cast<float>(sigma_),
            nullptr
        );
        
        manager_.synchronize();
        
        // Copy wavefunction back
        std::vector<float> wavefunction_real(grid_size_);
        std::vector<float> wavefunction_imag(grid_size_);
        manager_.copy_to_host(gpu_wavefunction_real_, wavefunction_real.data(),
                             grid_size_ * sizeof(float));
        manager_.copy_to_host(gpu_wavefunction_imag_, wavefunction_imag.data(),
                             grid_size_ * sizeof(float));
        
        // Create Wavefunction object
        Wavefunction psi(grid_size_, L_);
        // Use compute_squeezed_coherent to set values properly
        psi.compute_squeezed_coherent(alpha, beta, gamma, sigma_);
        
        // Cleanup
        manager_.free(gpu_alpha);
        manager_.free(gpu_beta);
        manager_.free(gpu_gamma);
        
        return {psi};
#else
        throw std::runtime_error("GPU not available");
#endif
    }
    
    std::vector<std::vector<Wavefunction>> encode_batch_gpu(const std::vector<Tensor>& features_batch) {
#ifdef USE_HIP
        int batch_size = features_batch.size();
        
        // Flatten all features
        std::vector<float> features_flat;
        features_flat.reserve(batch_size * input_dim_);
        
        for (const auto& features : features_batch) {
            Tensor flat = features;
            if (features.ndim() > 1) {
                flat = features.reshape({features.size()});
            }
            for (size_t i = 0; i < flat.size() && i < static_cast<size_t>(input_dim_); ++i) {
                features_flat.push_back(static_cast<float>(flat[i]));
            }
            // Pad if needed
            for (size_t i = flat.size(); i < static_cast<size_t>(input_dim_); ++i) {
                features_flat.push_back(0.0f);
            }
        }
        
        // Allocate GPU memory
        size_t features_size = batch_size * input_dim_ * sizeof(float);
        size_t alpha_size = batch_size * sizeof(float);
        size_t beta_size = batch_size * sizeof(float);
        size_t gamma_size = batch_size * sizeof(float);
        size_t wavefunction_size = batch_size * grid_size_ * sizeof(float);
        
        void* gpu_features = manager_.allocate(features_size);
        void* gpu_alpha = manager_.allocate(alpha_size);
        void* gpu_beta = manager_.allocate(beta_size);
        void* gpu_gamma = manager_.allocate(gamma_size);
        void* gpu_wf_real = manager_.allocate(wavefunction_size);
        void* gpu_wf_imag = manager_.allocate(wavefunction_size);
        
        // Copy features to GPU
        manager_.copy_to_device(features_flat.data(), gpu_features, features_size);
        
        // Get Chebyshev KAN coefficients from GPU layers
        // Each Chebyshev KAN has: input_dim * (chebyshev_order + 1) coefficients
        int chebyshev_order = 8;
        size_t coeffs_size = input_dim_ * (chebyshev_order + 1);
        
        std::vector<float> alpha_coeffs = get_chebyshev_coeffs(gpu_alpha_extractor_.get(), coeffs_size);
        std::vector<float> beta_coeffs = get_chebyshev_coeffs(gpu_beta_extractor_.get(), coeffs_size);
        std::vector<float> gamma_coeffs = get_chebyshev_coeffs(gpu_gamma_extractor_.get(), coeffs_size);
        
        void* gpu_alpha_coeffs = manager_.allocate(alpha_coeffs.size() * sizeof(float));
        void* gpu_beta_coeffs = manager_.allocate(beta_coeffs.size() * sizeof(float));
        void* gpu_gamma_coeffs = manager_.allocate(gamma_coeffs.size() * sizeof(float));
        
        manager_.copy_to_device(alpha_coeffs.data(), gpu_alpha_coeffs, alpha_coeffs.size() * sizeof(float));
        manager_.copy_to_device(beta_coeffs.data(), gpu_beta_coeffs, beta_coeffs.size() * sizeof(float));
        manager_.copy_to_device(gamma_coeffs.data(), gpu_gamma_coeffs, gamma_coeffs.size() * sizeof(float));
        
        // Launch batched quantum embedding kernel
        GPU::launch_batched_quantum_embedding(
            static_cast<const float*>(gpu_features),
            static_cast<const float*>(gpu_alpha_coeffs),
            static_cast<const float*>(gpu_beta_coeffs),
            static_cast<const float*>(gpu_gamma_coeffs),
            static_cast<float*>(gpu_alpha),
            static_cast<float*>(gpu_beta),
            static_cast<float*>(gpu_gamma),
            batch_size,
            input_dim_,
            8,  // chebyshev_order
            nullptr
        );
        
        // Launch wavefunction computation for batch
        GPU::launch_compute_squeezed_coherent(
            static_cast<const float*>(gpu_alpha),
            static_cast<const float*>(gpu_beta),
            static_cast<const float*>(gpu_gamma),
            static_cast<float*>(gpu_wf_real),
            static_cast<float*>(gpu_wf_imag),
            batch_size,
            grid_size_,
            static_cast<float>(L_),
            static_cast<float>(sigma_),
            nullptr
        );
        
        manager_.synchronize();
        
        // Copy results back
        std::vector<float> alpha_out(batch_size);
        std::vector<float> beta_out(batch_size);
        std::vector<float> gamma_out(batch_size);
        std::vector<float> wf_real(batch_size * grid_size_);
        std::vector<float> wf_imag(batch_size * grid_size_);
        
        manager_.copy_to_host(gpu_alpha, alpha_out.data(), alpha_size);
        manager_.copy_to_host(gpu_beta, beta_out.data(), beta_size);
        manager_.copy_to_host(gpu_gamma, gamma_out.data(), gamma_size);
        manager_.copy_to_host(gpu_wf_real, wf_real.data(), wavefunction_size);
        manager_.copy_to_host(gpu_wf_imag, wf_imag.data(), wavefunction_size);
        
        // Create Wavefunction objects
        std::vector<std::vector<Wavefunction>> results;
        for (int b = 0; b < batch_size; ++b) {
            Wavefunction psi(grid_size_, L_);
            // Use compute_squeezed_coherent with extracted parameters
            psi.compute_squeezed_coherent(alpha_out[b], beta_out[b], gamma_out[b], sigma_);
            results.push_back({psi});
        }
        
        // Cleanup
        manager_.free(gpu_features);
        manager_.free(gpu_alpha);
        manager_.free(gpu_beta);
        manager_.free(gpu_gamma);
        manager_.free(gpu_wf_real);
        manager_.free(gpu_wf_imag);
        manager_.free(gpu_alpha_coeffs);
        manager_.free(gpu_beta_coeffs);
        manager_.free(gpu_gamma_coeffs);
        
        return results;
#else
        throw std::runtime_error("GPU not available");
#endif
    }
    
    std::vector<float> get_chebyshev_coeffs(GPUKANLayer* layer, size_t expected_size) {
        if (!layer) return std::vector<float>(expected_size, 0.0f);
        auto params = layer->parameters();
        std::vector<float> coeffs(params.begin(), params.end());
        // Pad or truncate to expected size
        if (coeffs.size() < expected_size) {
            coeffs.resize(expected_size, 0.0f);
        } else if (coeffs.size() > expected_size) {
            coeffs.resize(expected_size);
        }
        return coeffs;
    }
};

