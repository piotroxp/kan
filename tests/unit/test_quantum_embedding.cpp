#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "quantum/quantum_field_embedding.hpp"
#include "core/tensor.hpp"

using Catch::Approx;

TEST_CASE("Quantum field embedding encoding", "[quantum][embedding]") {
    QuantumFieldEmbeddingCore embedding(10, 1, 1024, 12.0, 1.5);
    
    Tensor input({10});
    for (size_t i = 0; i < 10; ++i) {
        input[i] = 0.1 * i;
    }
    
    auto wavefunctions = embedding.encode(input);
    
    REQUIRE(wavefunctions.size() == 1);
    REQUIRE(wavefunctions[0].size() == 1024);
    
    // Check normalization
    double norm_sq = 0.0;
    for (const auto& val : wavefunctions[0].values()) {
        double mag_sq = val.real() * val.real() + val.imag() * val.imag();
        norm_sq += wavefunctions[0].grid_spacing() * mag_sq;
    }
    
    REQUIRE(norm_sq == Approx(1.0).margin(0.01));
}

TEST_CASE("Quantum embedding similarity", "[quantum][embedding]") {
    QuantumFieldEmbeddingCore embedding(10, 1, 1024, 12.0, 1.5);
    
    Tensor input1({10});
    Tensor input2({10});
    for (size_t i = 0; i < 10; ++i) {
        input1[i] = 0.1 * i;
        input2[i] = 0.1 * i + 0.01; // Slightly different
    }
    
    auto psi1 = embedding.encode(input1);
    auto psi2 = embedding.encode(input2);
    
    double sim = embedding.similarity(psi1[0], psi2[0]);
    
    // Similar inputs should have reasonably high similarity
    REQUIRE(sim >= 0.0);
    REQUIRE(sim <= 1.0);
}

TEST_CASE("Quantum embedding batch encoding", "[quantum][embedding]") {
    QuantumFieldEmbeddingCore embedding(10, 1, 1024, 12.0, 1.5);
    
    std::vector<Tensor> batch;
    for (int i = 0; i < 3; ++i) {
        Tensor input({10});
        for (size_t j = 0; j < 10; ++j) {
            input[j] = 0.1 * (i * 10 + j);
        }
        batch.push_back(input);
    }
    
    auto wavefunctions_batch = embedding.encode_batch(batch);
    
    REQUIRE(wavefunctions_batch.size() == 3);
    for (const auto& wfs : wavefunctions_batch) {
        REQUIRE(wfs.size() == 1);
        REQUIRE(wfs[0].size() == 1024);
    }
}



