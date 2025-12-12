#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <numeric>

class Tensor {
public:
    Tensor() = default;
    
    Tensor(std::vector<size_t> shape) 
        : shape_(shape), data_(compute_size(shape)) {}
    
    Tensor(std::vector<size_t> shape, double value)
        : shape_(shape), data_(compute_size(shape), value) {}
    
    Tensor(std::vector<size_t> shape, const std::vector<double>& data)
        : shape_(shape), data_(data) {
        if (data.size() != compute_size(shape)) {
            throw std::runtime_error("Tensor data size mismatch");
        }
    }
    
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }
    
    double& operator[](size_t idx) { return data_[idx]; }
    const double& operator[](size_t idx) const { return data_[idx]; }
    
    double& at(const std::vector<size_t>& indices) {
        return data_[flatten_index(indices)];
    }
    
    const double& at(const std::vector<size_t>& indices) const {
        return data_[flatten_index(indices)];
    }
    
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    
    std::vector<double>& values() { return data_; }
    const std::vector<double>& values() const { return data_; }
    
    void fill(double value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    double sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }
    
    double mean() const {
        return data_.empty() ? 0.0 : sum() / data_.size();
    }
    
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = compute_size(new_shape);
        if (new_size != data_.size()) {
            throw std::runtime_error("Cannot reshape: size mismatch");
        }
        Tensor result(new_shape);
        result.data_ = data_;
        return result;
    }
    
private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
    
    static size_t compute_size(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1UL, 
                              std::multiplies<size_t>());
    }
    
    size_t flatten_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Index dimension mismatch");
        }
        size_t idx = 0;
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::runtime_error("Index out of bounds");
            }
            idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return idx;
    }
};

