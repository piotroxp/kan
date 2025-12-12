#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "training/loss.hpp"
#include "training/optimizer.hpp"
#include "core/tensor.hpp"
#include "quantum/wavefunction.hpp"

using Catch::Approx;

TEST_CASE("Multi-task loss computation", "[training][loss]") {
    MultiTaskLoss loss_fn;
    
    SECTION("Audio reconstruction loss") {
        Tensor pred({10});
        Tensor target({10});
        
        for (size_t i = 0; i < 10; ++i) {
            pred[i] = 0.5;
            target[i] = 1.0;
        }
        
        double loss = loss_fn.audio_reconstruction_loss(pred, target);
        REQUIRE(loss == Approx(0.25));  // (0.5)^2
    }
    
    SECTION("Classification loss") {
        Tensor logits({3});
        Tensor labels({3});
        
        logits[0] = 2.0;  // High confidence positive
        logits[1] = -2.0; // High confidence negative
        logits[2] = 0.0;  // Neutral
        
        labels[0] = 1.0;
        labels[1] = 0.0;
        labels[2] = 1.0;
        
        double loss = loss_fn.classification_loss(logits, labels);
        REQUIRE(loss > 0.0);
        REQUIRE(std::isfinite(loss));
    }
    
    SECTION("Quantum fidelity loss") {
        Wavefunction psi1(1024, 12.0);
        Wavefunction psi2(1024, 12.0);
        
        psi1.compute_squeezed_coherent(0.0, 1.0, 0.0, 1.5);
        psi2.compute_squeezed_coherent(0.1, 1.0, 0.0, 1.5);  // Similar
        
        std::vector<Wavefunction> embeddings = {psi1, psi2};
        std::vector<std::pair<size_t, size_t>> positive_pairs = {{0, 1}};
        std::vector<std::pair<size_t, size_t>> negative_pairs;
        
        double loss = loss_fn.quantum_fidelity_loss(embeddings, positive_pairs, negative_pairs);
        REQUIRE(loss >= 0.0);
        REQUIRE(std::isfinite(loss));
    }
    
    SECTION("Normalization penalty") {
        Wavefunction psi(1024, 12.0);
        psi.compute_squeezed_coherent(0.0, 1.0, 0.0, 1.5);
        
        std::vector<Wavefunction> embeddings = {psi};
        double penalty = loss_fn.normalization_penalty(embeddings);
        
        // Should be small since wavefunction is normalized
        REQUIRE(penalty >= 0.0);
        REQUIRE(penalty < 0.1);
    }
}

TEST_CASE("AdamW optimizer", "[training][optimizer]") {
    SECTION("Initialize and step") {
        AdamW optimizer(0.01, 0.9, 0.999, 1e-8, 1e-5);
        
        std::vector<std::vector<double>> parameters = {{1.0, 2.0}, {3.0}};
        std::vector<std::vector<double>> gradients = {{-0.1, -0.2}, {-0.3}};
        
        optimizer.initialize(parameters);
        optimizer.step(parameters, gradients);
        
        // Parameters should be updated
        REQUIRE(parameters[0][0] != 1.0);
        REQUIRE(parameters[0][1] != 2.0);
        REQUIRE(parameters[1][0] != 3.0);
    }
    
    SECTION("Learning rate scheduling") {
        CosineAnnealingScheduler scheduler(1e-3, 1e-5, 1000);
        
        double lr_start = scheduler.get_lr(0);
        double lr_mid = scheduler.get_lr(500);
        double lr_end = scheduler.get_lr(1000);
        
        REQUIRE(lr_start == Approx(1e-3));
        REQUIRE(lr_end == Approx(1e-5));
        REQUIRE(lr_mid > lr_end);
        REQUIRE(lr_mid < lr_start);
    }
}



