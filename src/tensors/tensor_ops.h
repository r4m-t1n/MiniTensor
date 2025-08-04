#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <iostream>
#include <stdexcept>
#include "tensor.h"

template<typename T>
void check_tensor_validity(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.ndim != b.ndim) {
        throw std::invalid_argument(
            "ERROR: Shapes are not the same: " +
            std::to_string(n.ndim) + " and " + std::to_string(b.ndim)
        );
    }
    for (int i = 0; i < a.ndim; ++i) {
        if (a.shape[i] != b.shape[i]){
            throw std::invalid_argument(
                "ERROR: Shapes are not the same at dimension " + std::to_string(i) + ": " +
                std::to_string(a.shape[i]) + " and " + std::to_string(b.shape[i])
            );
        }
    }
}

template<typename T>
Tensor<T> tensor_add(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape);

    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> tensor_sub(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> tensor_mul(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}

template<typename T>
Tensor<T> tensor_div(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape);

    for (int i = 0; i < a.size; ++i) {
        if (b.data[i] == static_cast<T>(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
}


template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_add(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + static_cast<T>(scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_sub(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - static_cast<T>(scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_sub(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = static_cast<T>(scalar) - a.data[i];
    }
    return result;
}


template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_mul(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] * static_cast<T>(scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_div(const Tensor<T>& a, ScalarType scalar) {
    if (static_cast<T>(scalar) == T(0)) {
        throw std::runtime_error("ERROR: Division by zero is not allowed");
    }
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] / static_cast<T>(scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_div(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape);
    for (int i = 0; i < a.size; ++i) {
        if (a.data[i] == T(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = static_cast<T>(scalar) / a.data[i];
    }
    return result;
}


#endif