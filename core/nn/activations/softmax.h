#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <memory>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "tensors/tensor_math.h"
#include "tensors/tensor_reductions.h"

template<typename T>
std::shared_ptr<Tensor<T>> softmax(std::shared_ptr<Tensor<T>> tensor, int axis = -1) {
    if(axis != -1 && axis != tensor->ndim - 1) {
        throw std::runtime_error("ERROR: Softmax currently supports only the last axis");
    }

    auto max_tensor = max(tensor, axis);

    auto shifted = tensor_sub(tensor, max_tensor);

    auto exp_tensor = tensor_exp<T, T>(shifted);

    auto sum_tensor = std::make_shared<Tensor<T>>(tensor->shape, tensor->requires_grad);
    for(int i = 0; i < tensor->shape[0]; ++i) { // i will replace this with sum() later
        T row_sum = 0;
        for(int j = 0; j < tensor->shape[1]; ++j) {
            int idx = i * tensor->shape[1] + j;
            row_sum += exp_tensor->data[idx];
        }
        for(int j = 0; j < tensor->shape[1]; ++j) {
            int idx = i * tensor->shape[1] + j;
            sum_tensor->data[idx] = row_sum;
        }
    }

    auto result = tensor_div(exp_tensor, sum_tensor);

    return result;
}

#endif