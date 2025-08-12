#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional> 
#include <type_traits>

template<typename T>
class Tensor {
public:
    T* data;
    std::vector<int> shape;
    int ndim;
    int size;
    std::vector<int> stride;

    static std::vector<int> compute_stride(const std::vector<int>& shape, const int ndim) {
        std::vector<int> stride(ndim);
        int acc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            stride[i] = acc;
            acc *= shape[i];
        }
        return stride;
    }

    Tensor(const std::vector<int>& shape)
        : shape(shape), ndim(shape.size()), data(nullptr) {
        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) {
            throw std::invalid_argument("ERROR: Dimension must be positive.");
        }

        stride = compute_stride(shape, ndim);

        data = new T[size];
    }

    Tensor(const std::vector<T>& data_vec, const std::vector<int>& shape)
        : shape(shape), ndim(shape.size()), data(nullptr) {
        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }

        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) {
            throw std::invalid_argument("ERROR: Dimension must be positive.");
        }

        if (data_vec.size() != static_cast<size_t>(size)) {
            throw std::invalid_argument("ERROR: Data size does not match shape size.");
        }

        stride = compute_stride(shape, ndim);

        data = new T[size];
        if (!data) {
            throw std::bad_alloc();
        }

        for (int i = 0; i < size; ++i) {
            data[i] = data_vec[i];
        }
    }
    
    Tensor() = delete;

    Tensor(Tensor&& other) noexcept
        : data(other.data),
        shape(std::move(other.shape)),
        ndim(other.ndim),
        size(other.size),
        stride(std::move(other.stride)){
        other.data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            shape = std::move(other.shape);
            ndim = other.ndim;
            size = other.size;
            stride = std::move(other.stride);
            other.data = nullptr;
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor() {
        delete[] data;
    }
};

template<typename T>
std::vector<T> get_tensor_data(const Tensor<T>& tensor) {
    std::vector<T> data_vec(tensor.size);
    for (int i = 0; i < tensor.size; ++i) {
        data_vec[i] = tensor.data[i];
    }
    return data_vec;
}

#endif