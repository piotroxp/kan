#pragma once

#include "catch2/catch_approx.hpp"

#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace catch2_internal {

struct TestCase {
    const char* name;
    std::function<void()> func;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> func) {
        registry().push_back({name, std::move(func)});
    }
};

class TestFailure : public std::exception {
public:
    explicit TestFailure(std::string message) : message_(std::move(message)) {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

}  // namespace catch2_internal

#define CATCH2_DETAIL_CONCAT_INNER(a, b) a##b
#define CATCH2_DETAIL_CONCAT(a, b) CATCH2_DETAIL_CONCAT_INNER(a, b)

#define TEST_CASE(name, ...)                                                     \
    static void CATCH2_DETAIL_CONCAT(test_case_, __LINE__)();                      \
    static catch2_internal::Registrar CATCH2_DETAIL_CONCAT(                        \
        test_registrar_, __LINE__)(name, CATCH2_DETAIL_CONCAT(test_case_, __LINE__)); \
    static void CATCH2_DETAIL_CONCAT(test_case_, __LINE__)()

#define SECTION(name) for (bool section_once = true; section_once; section_once = false)

#define REQUIRE(expression)                                                      \
    do {                                                                          \
        if (!(expression)) {                                                     \
            std::ostringstream oss;                                              \
            oss << "Requirement failed: " << #expression                         \
                << " at " << __FILE__ << ":" << __LINE__;                       \
            throw catch2_internal::TestFailure(oss.str());                       \
        }                                                                         \
    } while (false)

namespace Catch {
using Approx = ::Catch::Approx;
}
