#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensor.h"
#include "tensor_broadcast.h"
#include "autograd/autograd_ops.h"

template<typename T>
void check_tensor_validity(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (a->ndim != b->ndim) {
        throw std::invalid_argument("ERROR: Shapes are not the same dimension");
    }
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]){
            throw std::invalid_argument("ERROR: Shapes are not the same at dimension " + std::to_string(i));
        }
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> tensor_add(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(*a, *b);
    auto result_data = std::vector<T>(a_broadcasted.size);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result_data[i] = a_broadcasted.data[i] + b_broadcasted.data[i];
    }
    auto result = std::make_shared<Tensor<T>>(result_data, a_broadcasted.shape, a->requires_grad || b->requires_grad);
    if (result->requires_grad) {
        result->parents = {a, b};
        result->grad_fn = std::make_unique<AddBackward<T>>(a, b);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> tensor_sub(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(*a, *b);
    auto result_data = std::vector<T>(a_broadcasted.size);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result_data[i] = a_broadcasted.data[i] - b_broadcasted.data[i];
    }
    auto result = std::make_shared<Tensor<T>>(result_data, a_broadcasted.shape, a->requires_grad || b->requires_grad);
    if (result->requires_grad) {
        result->parents = {a, b};
        result->grad_fn = std::make_unique<SubBackward<T>>(a, b);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> tensor_mul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(*a, *b);
    auto result_data = std::vector<T>(a_broadcasted.size);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        result_data[i] = a_broadcasted.data[i] * b_broadcasted.data[i];
    }
    auto result = std::make_shared<Tensor<T>>(result_data, a_broadcasted.shape, a->requires_grad || b->requires_grad);
    if (result->requires_grad) {
        result->parents = {a, b};
        result->grad_fn = std::make_unique<MulBackward<T>>(a, b);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> tensor_div(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    auto [a_broadcasted, b_broadcasted] = broadcast(*a, *b);
    auto result_data = std::vector<T>(a_broadcasted.size);
    for (int i = 0; i < a_broadcasted.size; ++i) {
        if (b_broadcasted.data[i] == static_cast<T>(0)) throw std::runtime_error("ERROR: Division by zero");
        result_data[i] = a_broadcasted.data[i] / b_broadcasted.data[i];
    }
    auto result = std::make_shared<Tensor<T>>(result_data, a_broadcasted.shape, a->requires_grad || b->requires_grad);
    if (result->requires_grad) {
        result->parents = {a, b};
        result->grad_fn = std::make_unique<DivBackward<T>>(a, b);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> tensor_scalar_add(const std::shared_ptr<Tensor<T>>& a, U scalar) {
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) result_data[i] = a->data[i] + static_cast<T>(scalar);
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<AddScalarBackward<T>>(a);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> tensor_scalar_sub(const std::shared_ptr<Tensor<T>>& a, U scalar) {
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) result_data[i] = a->data[i] - static_cast<T>(scalar);
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<SubScalarBackward<T>>(a);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> scalar_tensor_sub(U scalar, const std::shared_ptr<Tensor<T>>& a) {
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) result_data[i] = static_cast<T>(scalar) - a->data[i];
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<ScalarTensorSubBackward<T, U>>(a);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> tensor_scalar_mul(const std::shared_ptr<Tensor<T>>& a, U scalar) {
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) result_data[i] = a->data[i] * static_cast<T>(scalar);
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<MulScalarBackward<T, U>>(a, scalar);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> tensor_scalar_div(const std::shared_ptr<Tensor<T>>& a, U scalar) {
    if (static_cast<T>(scalar) == 0) throw std::runtime_error("ERROR: Division by zero");
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) result_data[i] = a->data[i] / static_cast<T>(scalar);
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<DivScalarBackward<T, U>>(a, scalar);
    }
    return result;
}

template<typename T, typename U>
std::shared_ptr<Tensor<T>> scalar_tensor_div(U scalar, const std::shared_ptr<Tensor<T>>& a) {
    auto result_data = std::vector<T>(a->size);
    for (int i = 0; i < a->size; ++i) {
        if (a->data[i] == static_cast<T>(0)) throw std::runtime_error("ERROR: Division by zero");
        result_data[i] = static_cast<T>(scalar) / a->data[i];
    }
    auto result = std::make_shared<Tensor<T>>(result_data, a->shape, a->requires_grad);
    if (result->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<ScalarTensorDivBackward<T, U>>(scalar, a);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& a) {
    if (a->ndim != 2) throw std::invalid_argument("ERROR: Transpose is only for 2D tensors.");
    
    std::vector<int> new_shape = {a->shape[1], a->shape[0]};
    auto result = std::make_shared<Tensor<T>>(new_shape, a->requires_grad);

    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[1]; j++) {
            result->data[j * result->stride[0] + i] = a->data[i * a->stride[0] + j];
        }
    }
    if (a->requires_grad) {
        result->parents = {a};
        result->grad_fn = std::make_unique<TransposeBackward<T>>(a);
    }
    
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> mat_mul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (a->ndim != 2 || b->ndim != 2) throw std::invalid_argument("ERROR: Both tensors must be 2D matrices");
    if (a->shape[1] != b->shape[0]) throw std::invalid_argument("ERROR: Shapes are not valid to multiply");
    auto result = std::make_shared<Tensor<T>>(std::vector<int>{a->shape[0], b->shape[1]}, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            T sum_val = 0;
            for (int r = 0; r < a->shape[1]; r++) {
                sum_val += a->data[i * a->stride[0] + r] * b->data[r * b->stride[0] + j];
            }
            result->data[i * result->stride[0] + j] = sum_val;
        }
    }
    if (result->requires_grad) {
        result->parents = {a, b};
        result->grad_fn = std::make_unique<MatMulBackward<T>>(a, b);
    }
    return result;
}

#endif