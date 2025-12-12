#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "core/bspline_kan.hpp"
#include "core/chebyshev_kan.hpp"
#include "core/sinc_kan.hpp"
#include "core/fourier_kan.hpp"
#include "core/rbf_kan.hpp"
#include "core/piecewise_linear_kan.hpp"
#include "core/tensor.hpp"
#include <cmath>

using Catch::Approx;

TEST_CASE("B-spline KAN layer", "[kan][bspline]") {
    BSplineKANLayer layer(2, 3, 5, 3);
    
    // Initialize with small random values
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1 * (i % 10);
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    // Output should be finite
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}

TEST_CASE("Chebyshev KAN layer", "[kan][chebyshev]") {
    ChebyshevKANLayer layer(2, 3, 8, 8);
    
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1;
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}

TEST_CASE("Sinc KAN layer", "[kan][sinc]") {
    SincKANLayer layer(2, 3, 5);
    
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1;
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}

TEST_CASE("Fourier KAN layer", "[kan][fourier]") {
    FourierKANLayer layer(2, 3, 9); // 9 = 4 modes * 2 + 1 DC
    
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1;
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}

TEST_CASE("RBF KAN layer", "[kan][rbf]") {
    RBFKANLayer layer(2, 3, 5, 1.0);
    
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1;
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}

TEST_CASE("Piecewise Linear KAN layer", "[kan][piecewise]") {
    PiecewiseLinearKANLayer layer(2, 3, 5);
    
    auto& params = layer.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1 * i;
    }
    
    Tensor input({2});
    input[0] = 0.5;
    input[1] = -0.5;
    
    Tensor output = layer.forward(input);
    
    REQUIRE(output.shape()[0] == 3);
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(std::isfinite(output[i]));
    }
}


