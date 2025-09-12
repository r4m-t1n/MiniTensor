#ifndef TENSOR_BROADCAST_H
#define TENSOR_BROADCAST_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include "tensor.h"

inline std::vector<int> broadcast_shape(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
    int ndim_a = shape_a.size();
    int ndim_b = shape_b.size();
    int max_ndim = std::max(ndim_a, ndim_b);
    std::vector<int> result_shape(max_ndim);

    for (int i = 1; i <= max_ndim; ++i) {
        int dim_a = (i <= ndim_a) ? shape_a[ndim_a - i] : 1;
        int dim_b = (i <= ndim_b) ? shape_b[ndim_b - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::invalid_argument("ERROR: Tensors are not broadcastable.");
        }
        result_shape[max_ndim - i] = std::max(dim_a, dim_b);
    }
    return result_shape;
}

template<typename T>
Tensor<T> expand_tensor(const Tensor<T>& tensor, const std::vector<int>& target_shape) {
    Tensor<T> result(target_shape, tensor.requires_grad);
    int target_ndim = target_shape.size();
    int tensor_ndim = tensor.ndim;

    std::vector<int> broadcast_strides(target_ndim, 0);
    int shape_diff = target_ndim - tensor_ndim;
    for (int i = 0; i < tensor_ndim; ++i) {
        if (tensor.shape[i] == 1) {
            broadcast_strides[i + shape_diff] = 0;
        } else {
            broadcast_strides[i + shape_diff] = tensor.stride[i];
        }
    }

    for (int i = 0; i < result.size; ++i) {
        int original_index = 0;
        int temp_i = i;
        for (int j = target_ndim - 1; j >= 0; --j) {
            int current_dim_size = result.shape[j];
            int coord = temp_i % current_dim_size;
            temp_i /= current_dim_size;
            original_index += coord * broadcast_strides[j];
        }
        result.data[i] = tensor.data[original_index];
    }

    return result;
}

template<typename T>
std::pair<Tensor<T>, Tensor<T>> broadcast(const Tensor<T>& a, const Tensor<T>& b) {
    std::vector<int> result_shape = broadcast_shape(a.shape, b.shape);
    Tensor<T> a_broadcasted = (a.shape == result_shape) ? Tensor<T>(
        std::vector<T>(a.data, a.data + a.size), a.shape) : expand_tensor(a, result_shape);
    Tensor<T> b_broadcasted = (b.shape == result_shape) ? Tensor<T>(
        std::vector<T>(b.data, b.data + b.size), b.shape) : expand_tensor(b, result_shape);
    
    return { std::move(a_broadcasted), std::move(b_broadcasted) };
}

template<typename T>
Tensor<T> unbroadcast(const Tensor<T>& grad, const std::vector<int>& target_shape) {
    auto current_grad = std::make_shared<Tensor<T>>(std::vector<T>(grad.data, grad.data + grad.size), grad.shape);

    while (current_grad->ndim > target_shape.size()) {
        current_grad = sum(current_grad, 0);
    }

    for (int i = 0; i < current_grad->ndim; ++i) {
        if (current_grad->shape[i] > target_shape[i]) {
            current_grad = sum(current_grad, i);
        }
    }
    return Tensor<T>(std::vector<T>(current_grad->data, current_grad->data + current_grad->size), current_grad->shape);
}

#endif