#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <iostream>
#include <stdexcept>
#include "tensor.h"
#include <vector>
#include <cmath>
#include "tensors/tensor_broadcast.h"
#include "autograd/autograd_ops.h"
#include "autograd/autograd_reduction.h"

template<typename T>
void check_tensor_validity(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.ndim != b.ndim) {
        throw std::invalid_argument(
            "ERROR: Shapes are not the same: " +
            std::to_string(a.ndim) + " and " + std::to_string(b.ndim)
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
    auto [a_broadcasted, b_broadcasted] = broadcast(a, b);
    Tensor<T> result(a_broadcasted.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result.data[i] = a_broadcasted.data[i] + b_broadcasted.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<AddBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}
template<typename T>
Tensor<T> tensor_sub(const Tensor<T>& a, const Tensor<T>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(a, b);
    Tensor<T> result(a_broadcasted.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result.data[i] = a_broadcasted.data[i] - b_broadcasted.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<SubBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T>
Tensor<T> tensor_mul(const Tensor<T>& a, const Tensor<T>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(a, b);
    Tensor<T> result(a_broadcasted.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result.data[i] = a_broadcasted.data[i] * b_broadcasted.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<MulBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T>
Tensor<T> tensor_div(const Tensor<T>& a, const Tensor<T>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(a, b);
    Tensor<T> result(a_broadcasted.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        if (b_broadcasted.data[i] == static_cast<T>(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = a_broadcasted.data[i] / b_broadcasted.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<DivBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_add(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<AddScalarBackward<T>>(const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_sub(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<SubScalarBackward<T>>(const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_sub(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = static_cast<T>(scalar) - a.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<ScalarTensorSubBackward<T, ScalarType>>(scalar, const_cast<Tensor<T>*>(&a));
    }
    return result;
}


template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_mul(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] * static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<MulScalarBackward<T, ScalarType>>(const_cast<Tensor<T>*>(&a), scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_div(const Tensor<T>& a, ScalarType scalar) {
    if (static_cast<T>(scalar) == T(0)) {
        throw std::runtime_error("ERROR: Division by zero is not allowed");
    }
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] / static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<DivScalarBackward<T, ScalarType>>(const_cast<Tensor<T>*>(&a), scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_div(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        if (a.data[i] == T(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = static_cast<T>(scalar) / a.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<ScalarTensorDivBackward<T, ScalarType>>(scalar, const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_add(t1, t2);
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_sub(t1, t2);
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_mul(t1, t2);
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_div(t1, t2);
}

template<typename T, typename U>
Tensor<T> operator+(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_add(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator-(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_sub(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator*(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_mul(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator/(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_div(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator+(U scalar, const Tensor<T>& tensor) {
    return tensor_scalar_add(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator-(U scalar, const Tensor<T>& tensor) {
    return scalar_tensor_sub(scalar, tensor); 
}

template<typename T, typename U>
Tensor<T> operator*(U scalar, const Tensor<T>& tensor) {
    return tensor_scalar_mul(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator/(U scalar, const Tensor<T>& tensor) {
    return scalar_tensor_div(scalar, tensor);
}

template<typename T>
Tensor<T> mat_mul(const Tensor<T> &a, const Tensor<T> &b){
    if (a.ndim != 2 || b.ndim != 2){
        throw std::invalid_argument("ERROR: Both tensors must be 2D matrices");
    }

    int m = a.shape[0];
    int k1 = a.shape[1];
    int k2 = b.shape[0];
    int n = b.shape[1];

    if (k1 != k2){
        throw std::invalid_argument(
            "ERROR: Shapes are not valid to multiply: " +
            std::to_string(m) + "x" + std::to_string(k1) +
            " and " +
            std::to_string(k2) + "x" + std::to_string(n)
        );
    }

    bool result_requires_grad = a.requires_grad || b.requires_grad;
    Tensor<T> result({m, n}, result_requires_grad);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T sum = 0;
            for (int r = 0; r < k1; r++) {
                int idx_a = i * a.stride[0] + r * a.stride[1];
                int idx_b = r * b.stride[0] + j * b.stride[1];
                sum += a.data[idx_a] * b.data[idx_b];
            }
            int idx_res = i * result.stride[0] + j * result.stride[1];
            result.data[idx_res] = sum;
        }
    }

    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<MatMulBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }

    return result;

}

template<typename T>
Tensor<T> transpose(const Tensor<T>& a) {
    if (a.ndim != 2) { // for now, I want to implement only for 2d tensors
        throw std::invalid_argument("ERROR: Transpose is currently only implemented for 2D tensors.");
    }

    std::vector<int> new_shape = {a.shape[1], a.shape[0]};

    Tensor<T> result(new_shape, a.requires_grad);

    for (int i=0; i<a.shape[0]; i++){
        for (int j=0; j<a.shape[1]; j++){
            int old_index = i * a.stride[0] + j * a.stride[1];
            int new_index = j * result.stride[0] + i * result.stride[1];
            result.data[new_index] = a.data[old_index];
        }
    }

    return result;

}

template<typename T>
Tensor<T> sum(const Tensor<T>& tensor, int axis = -1) {
    if (axis == -1) {
        T total_sum = 0;
        for (int i = 0; i < tensor.size; ++i) {
            total_sum += tensor.data[i];
        }
        return Tensor<T>({total_sum}, {1}, tensor.requires_grad);
    }

    if (axis < 0 || axis >= tensor.ndim) {
        throw std::invalid_argument("ERROR: Invalid axis for sum operation.");
    }

    std::vector<int> new_shape = tensor.shape;
    new_shape[axis] = 1;

    Tensor<T> result(new_shape, tensor.requires_grad);
    std::fill(result.data, result.data + result.size, static_cast<T>(0));

    for (int i = 0; i < tensor.size; ++i) {
        int result_index = 0;
        int original_index_temp = i;

        for (int dim = 0; dim < tensor.ndim; ++dim) {
            int coord = (original_index_temp / tensor.stride[dim]) % tensor.shape[dim];
            if (dim != axis) {
                result_index += coord * result.stride[dim];
            }
        }
        result.data[result_index] += tensor.data[i];
    }

    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&tensor));
        result.grad_fn = std::make_unique<SumBackward<T>>(const_cast<Tensor<T>*>(&tensor), axis);
    }

    return result;
}

#endif