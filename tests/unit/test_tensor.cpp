#include <catch2/catch_test_macros.hpp>
#include "core/tensor.hpp"

TEST_CASE("Tensor creation and basic operations", "[tensor]") {
    SECTION("Create tensor with shape") {
        Tensor t({2, 3});
        REQUIRE(t.shape().size() == 2);
        REQUIRE(t.shape()[0] == 2);
        REQUIRE(t.shape()[1] == 3);
        REQUIRE(t.size() == 6);
    }
    
    SECTION("Tensor indexing") {
        Tensor t({2, 3});
        t[0] = 1.0;
        t[1] = 2.0;
        REQUIRE(t[0] == 1.0);
        REQUIRE(t[1] == 2.0);
    }
    
    SECTION("Tensor at() with indices") {
        Tensor t({2, 3});
        t.at({0, 0}) = 5.0;
        t.at({1, 2}) = 10.0;
        REQUIRE(t.at({0, 0}) == 5.0);
        REQUIRE(t.at({1, 2}) == 10.0);
    }
    
    SECTION("Tensor fill and sum") {
        Tensor t({3, 4}, 2.0);
        REQUIRE(t.sum() == 24.0);
        REQUIRE(t.mean() == 2.0);
    }
    
    SECTION("Tensor reshape") {
        Tensor t({2, 3});
        for (size_t i = 0; i < 6; ++i) {
            t[i] = static_cast<double>(i);
        }
        Tensor reshaped = t.reshape({6});
        REQUIRE(reshaped.shape()[0] == 6);
        REQUIRE(reshaped[0] == 0.0);
        REQUIRE(reshaped[5] == 5.0);
    }
}


