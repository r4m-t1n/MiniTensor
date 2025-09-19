#ifndef TENSOR_REDUCTIONS_H
#define TENSOR_REDUCTIONS_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <limits>
#include "tensor.h"
#include "autograd/autograd_reductions.h"
#include "tensors/tensor_ops.h"


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
                multiplier *= (j > 0 && j-1 != axis) ? tensor->shape[j-1] : 1;
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

template<typename T>
std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>>& tensor, int axis = -1) {
    auto sum_res = sum(tensor, axis);
    int n = (axis == -1) ? tensor->size : tensor->shape[axis];
    auto result = tensor_scalar_div(sum_res, static_cast<T>(n));
    
    if (tensor->requires_grad) {
        result->parents = {tensor};
        result->grad_fn = std::make_unique<MeanBackward<T>>(tensor, axis);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> max(const std::shared_ptr<Tensor<T>>& tensor, int axis = -1) {
    if (axis < -1 || axis >= tensor->ndim) {
        throw std::invalid_argument("ERROR: Invalid axis for max operation.");
    }

    if (axis == -1) {
        T max_val = tensor->data[0];
        int max_idx = 0;
        for (int i = 1; i < tensor->size; ++i) {
            if (tensor->data[i] > max_val) {
                max_val = tensor->data[i];
                max_idx = i;
            }
        }
        auto result = std::make_shared<Tensor<T>>(std::vector<T>{max_val}, std::vector<int>{1}, tensor->requires_grad);
        if (tensor->requires_grad) {
            result->parents = {tensor};
            result->grad_fn = std::make_unique<MaxBackward<T>>(tensor, std::vector<int>{max_idx});
        }
        return result;
    }
    
    std::vector<int> result_shape;
    for(int i=0; i<tensor->ndim; ++i) {
        if (i != axis) {
            result_shape.push_back(tensor->shape[i]);
        }
    }
    if (result_shape.empty()) result_shape.push_back(1);
    
    auto result = std::make_shared<Tensor<T>>(result_shape, tensor->requires_grad);
    std::vector<int> max_indices(result->size);
    std::fill(result->data.get(), result->data.get() + result->size, std::numeric_limits<T>::lowest());

    if (axis == 0) {
        int outer_dim = tensor->shape[0];
        int inner_dim = tensor->size / outer_dim;
        for (int j = 0; j < inner_dim; ++j) {
            T max_val = tensor->data[j];
            int max_idx = j;
            for (int i = 1; i < outer_dim; ++i) {
                if (tensor->data[i * inner_dim + j] > max_val) {
                    max_val = tensor->data[i * inner_dim + j];
                    max_idx = i * inner_dim + j;
                }
            }
            result->data[j] = max_val;
            max_indices[j] = max_idx;
        }
    } else {
        throw std::runtime_error("ERROR: Max operation for axis other than 0 or -1 is not implemented.");
    }

    if (tensor->requires_grad) {
        result->parents = {tensor};
        result->grad_fn = std::make_unique<MaxBackward<T>>(tensor, max_indices);
    }
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> min(const std::shared_ptr<Tensor<T>>& tensor, int axis = -1) {
    if (axis < -1 || axis >= tensor->ndim) {
        throw std::invalid_argument("ERROR: Invalid axis for min operation.");
    }

    if (axis == -1) {
        T min_val = tensor->data[0];
        int min_idx = 0;
        for (int i = 1; i < tensor->size; ++i) {
            if (tensor->data[i] < min_val) {
                min_val = tensor->data[i];
                min_idx = i;
            }
        }
        auto result = std::make_shared<Tensor<T>>(std::vector<T>{min_val}, std::vector<int>{1}, tensor->requires_grad);
        if (tensor->requires_grad) {
            result->parents = {tensor};
            result->grad_fn = std::make_unique<MinBackward<T>>(tensor, std::vector<int>{min_idx});
        }
        return result;
    }

    std::vector<int> result_shape;
    for(int i=0; i<tensor->ndim; ++i) {
        if (i != axis) {
            result_shape.push_back(tensor->shape[i]);
        }
    }
    if (result_shape.empty()) result_shape.push_back(1);

    auto result = std::make_shared<Tensor<T>>(result_shape, tensor->requires_grad);
    std::vector<int> min_indices(result->size);
    std::fill(result->data.get(), result->data.get() + result->size, std::numeric_limits<T>::max());

    if (axis == 0) {
        int outer_dim = tensor->shape[0];
        int inner_dim = tensor->size / outer_dim;
        for (int j = 0; j < inner_dim; ++j) {
            T min_val = tensor->data[j];
            int min_idx = j;
            for (int i = 1; i < outer_dim; ++i) {
                if (tensor->data[i * inner_dim + j] < min_val) {
                    min_val = tensor->data[i * inner_dim + j];
                    min_idx = i * inner_dim + j;
                }
            }
            result->data[j] = min_val;
            min_indices[j] = min_idx;
        }
    } else {
        throw std::runtime_error("ERROR: Min operation for axis other than 0 or -1 is not implemented.");
    }

    if (tensor->requires_grad) {
        result->parents = {tensor};
        result->grad_fn = std::make_unique<MinBackward<T>>(tensor, min_indices);
    }
    return result;
}

#endif