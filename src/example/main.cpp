#include <iostream>
#include "core/tensor.hpp"
#include "core/bspline_kan.hpp"
#include "core/chebyshev_kan.hpp"
#include "core/sinc_kan.hpp"
#include "quantum/wavefunction.hpp"
#include "quantum/quantum_field_embedding.hpp"

int main() {
    std::cout << "=== KAN Speech Model Example ===\n\n";
    
    // Test Tensor
    std::cout << "1. Testing Tensor...\n";
    Tensor t({2, 3});
    t.fill(1.5);
    std::cout << "   Tensor shape: [" << t.shape()[0] << ", " << t.shape()[1] << "]\n";
    std::cout << "   Tensor sum: " << t.sum() << "\n\n";
    
    // Test B-spline KAN
    std::cout << "2. Testing B-spline KAN...\n";
    BSplineKANLayer bspline_kan(3, 2, 5, 3);
    auto& params = bspline_kan.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1 * (i % 10);
    }
    Tensor input({3});
    input[0] = 0.5;
    input[1] = -0.3;
    input[2] = 0.1;
    Tensor output = bspline_kan.forward(input);
    std::cout << "   Input: [" << input[0] << ", " << input[1] << ", " << input[2] << "]\n";
    std::cout << "   Output: [" << output[0] << ", " << output[1] << "]\n\n";
    
    // Test Chebyshev KAN
    std::cout << "3. Testing Chebyshev KAN...\n";
    ChebyshevKANLayer chebyshev_kan(3, 2, 8, 8);
    auto& cheb_params = chebyshev_kan.parameters();
    for (size_t i = 0; i < cheb_params.size(); ++i) {
        cheb_params[i] = 0.1;
    }
    Tensor cheb_output = chebyshev_kan.forward(input);
    std::cout << "   Output: [" << cheb_output[0] << ", " << cheb_output[1] << "]\n\n";
    
    // Test Wavefunction
    std::cout << "4. Testing Quantum Wavefunction...\n";
    Wavefunction psi(1024, 12.0);
    psi.compute_squeezed_coherent(0.5, 1.0, 0.0, 1.5);
    
    // Check normalization
    double norm_sq = 0.0;
    for (const auto& val : psi.values()) {
        double mag_sq = val.real() * val.real() + val.imag() * val.imag();
        norm_sq += psi.grid_spacing() * mag_sq;
    }
    std::cout << "   Wavefunction size: " << psi.size() << "\n";
    std::cout << "   Normalization: " << norm_sq << " (should be ~1.0)\n\n";
    
    // Test Quantum Field Embedding
    std::cout << "5. Testing Quantum Field Embedding...\n";
    QuantumFieldEmbeddingCore embedding(10, 1, 1024, 12.0, 1.5);
    
    Tensor audio_features({10});
    for (size_t i = 0; i < 10; ++i) {
        audio_features[i] = 0.1 * i;
    }
    
    auto wavefunctions = embedding.encode(audio_features);
    std::cout << "   Encoded " << wavefunctions.size() << " wavefunction(s)\n";
    std::cout << "   Wavefunction size: " << wavefunctions[0].size() << "\n";
    
    // Test similarity
    Tensor audio_features2({10});
    for (size_t i = 0; i < 10; ++i) {
        audio_features2[i] = 0.1 * i + 0.05;
    }
    auto wavefunctions2 = embedding.encode(audio_features2);
    double similarity = embedding.similarity(wavefunctions[0], wavefunctions2[0]);
    std::cout << "   Similarity between similar inputs: " << similarity << "\n\n";
    
    std::cout << "=== All tests completed successfully! ===\n";
    
    return 0;
}



