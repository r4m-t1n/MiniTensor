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
    if (tensor.shape == target_shape) {
        return tensor;
    }

    Tensor<T> result(target_shape, tensor.requires_grad);
    int target_ndim = target_shape.size();
    int tensor_ndim = tensor.ndim;

    std::vector<int> broadcast_strides(target_ndim, 0);
    for (int i = 1; i <= tensor_ndim; ++i) {
        if (tensor.shape[tensor_ndim - i] == target_shape[target_ndim - i]) {
            broadcast_strides[target_ndim - i] = tensor.stride[tensor_ndim - i];
        }
    }

    for (int i = 0; i < result.size; ++i) {
        int original_index = 0;
        int temp_i = i;
        for (int j = 0; j < target_ndim; ++j) {
            int coord = (temp_i / result.stride[j]) % target_shape[j];
            original_index += coord * broadcast_strides[j];
        }
        result.data[i] = tensor.data[original_index];
    }

    return result;
}

template<typename T>
std::pair<Tensor<T>, Tensor<T>> broadcast(const Tensor<T>& a, const Tensor<T>& b) {
    std::vector<int> result_shape = broadcast_shape(a.shape, b.shape);
    Tensor<T> a_broadcasted = expand_tensor(a, result_shape);
    Tensor<T> b_broadcasted = expand_tensor(b, result_shape);
    return {a_broadcasted, b_broadcasted};
}