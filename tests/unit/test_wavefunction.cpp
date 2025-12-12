#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "quantum/wavefunction.hpp"
#include <cmath>

using Catch::Approx;

TEST_CASE("Wavefunction creation", "[wavefunction]") {
    Wavefunction psi(1024, 12.0);
    
    REQUIRE(psi.size() == 1024);
    REQUIRE(psi.grid_spacing() == Approx(24.0 / 1024.0));
    REQUIRE(psi.grid_range() == 12.0);
}

TEST_CASE("Squeezed coherent state", "[wavefunction]") {
    Wavefunction psi(1024, 12.0);
    
    double alpha = 0.5;
    double beta = 1.0;
    double gamma = 0.0;
    double sigma = 1.5;
    
    psi.compute_squeezed_coherent(alpha, beta, gamma, sigma);
    
    // Check normalization
    double norm_sq = 0.0;
    for (const auto& val : psi.values()) {
        double mag_sq = val.real() * val.real() + val.imag() * val.imag();
        norm_sq += psi.grid_spacing() * mag_sq;
    }
    
    REQUIRE(norm_sq == Approx(1.0).margin(0.01));
}

TEST_CASE("Wavefunction inner product", "[wavefunction]") {
    Wavefunction psi1(1024, 12.0);
    Wavefunction psi2(1024, 12.0);
    
    psi1.compute_squeezed_coherent(0.0, 1.0, 0.0, 1.5);
    psi2.compute_squeezed_coherent(0.0, 1.0, 0.0, 1.5);
    
    // Same wavefunction should have fidelity = 1
    double fid = psi1.fidelity(psi2);
    REQUIRE(fid == Approx(1.0).margin(0.01));
}

TEST_CASE("Wavefunction fidelity for different states", "[wavefunction]") {
    Wavefunction psi1(1024, 12.0);
    Wavefunction psi2(1024, 12.0);
    
    // Similar states (small displacement)
    psi1.compute_squeezed_coherent(0.0, 1.0, 0.0, 1.5);
    psi2.compute_squeezed_coherent(0.1, 1.0, 0.0, 1.5);
    
    double fid = psi1.fidelity(psi2);
    REQUIRE(fid > 0.5); // Should be reasonably high
    
    // Very different states (large displacement)
    psi2.compute_squeezed_coherent(5.0, 1.0, 0.0, 1.5);
    fid = psi1.fidelity(psi2);
    REQUIRE(fid < 0.5); // Should be lower
}

TEST_CASE("Wavefunction normalization", "[wavefunction]") {
    Wavefunction psi(1024, 12.0);
    psi.compute_squeezed_coherent(1.0, 2.0, 0.5, 1.5);
    
    // Normalize again (should be idempotent)
    psi.normalize();
    
    double norm_sq = 0.0;
    for (const auto& val : psi.values()) {
        double mag_sq = val.real() * val.real() + val.imag() * val.imag();
        norm_sq += psi.grid_spacing() * mag_sq;
    }
    
    REQUIRE(norm_sq == Approx(1.0).margin(0.01));
}


