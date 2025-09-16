#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensor.h"
#include "tensor_broadcast.h"
#include "autograd/autograd_ops.h"
#include "autograd/autograd_reduction.h"

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

template<typename T>
std::shared_ptr<Tensor<T>> sum(const std::shared_ptr<Tensor<T>>& tensor, int axis = -1) {
    if (axis < -1 || axis >= tensor->ndim) {
        throw std::invalid_argument("ERROR: Invalid axis for sum operation.");
    }

    if (axis == -1) {
        T total_sum = 0;
        for (int i = 0; i < tensor->size; ++i) {
            total_sum += tensor->data[i];
        }
        auto result = std::make_shared<Tensor<T>>(std::vector<T>{total_sum}, std::vector<int>{1}, tensor->requires_grad);
        if (result->requires_grad) {
            result->parents = {tensor};
            result->grad_fn = std::make_unique<SumBackward<T>>(tensor, axis);
        }
        return result;
    }

    std::vector<int> result_shape;
    for (int i = 0; i < tensor->ndim; ++i) {
        if (i != axis) {
            result_shape.push_back(tensor->shape[i]);
        }
    }
    if (result_shape.empty()) {
        result_shape.push_back(1);
    }

    auto result = std::make_shared<Tensor<T>>(result_shape, tensor->requires_grad);
    std::fill(result->data.get(), result->data.get() + result->size, static_cast<T>(0));

    for(int i = 0; i < tensor->size; ++i) {
        int original_idx = i;
        int result_idx = 0;
        int multiplier = 1;
        for(int j = tensor->ndim - 1; j >= 0; --j) {
            int coord = original_idx % tensor->shape[j];
            original_idx /= tensor->shape[j];
            if(j != axis) {
                result_idx += coord * multiplier;
                multiplier *= tensor->shape[j];
            }
        }
        result->data[result_idx] += tensor->data[i];
    }
    
    if (result->requires_grad) {
        result->parents = {tensor};
        result->grad_fn = std::make_unique<SumBackward<T>>(tensor, axis);
    }
    return result;
}

#endif