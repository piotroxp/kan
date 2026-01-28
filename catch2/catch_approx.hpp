#pragma once

#include <cmath>

namespace Catch {

class Approx {
public:
    explicit Approx(double value) : value_(value), margin_(1e-12) {}

    Approx& margin(double margin) {
        margin_ = margin;
        return *this;
    }

    friend bool operator==(double lhs, const Approx& rhs) {
        return std::abs(lhs - rhs.value_) <= rhs.margin_;
    }

    friend bool operator==(const Approx& lhs, double rhs) {
        return rhs == lhs;
    }

    friend bool operator!=(double lhs, const Approx& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const Approx& lhs, double rhs) {
        return !(rhs == lhs);
    }

private:
    double value_;
    double margin_;
};

}  // namespace Catch
