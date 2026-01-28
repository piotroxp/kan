#include "catch2/catch_test_macros.hpp"

#include <iostream>

int main() {
    size_t failed = 0;
    for (const auto& test : catch2_internal::registry()) {
        try {
            test.func();
            std::cout << "[PASS] " << test.name << "\n";
        } catch (const std::exception& ex) {
            ++failed;
            std::cout << "[FAIL] " << test.name << ": " << ex.what() << "\n";
        } catch (...) {
            ++failed;
            std::cout << "[FAIL] " << test.name << ": unknown error\n";
        }
    }

    std::cout << "Tests run: " << catch2_internal::registry().size()
              << ", failures: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
