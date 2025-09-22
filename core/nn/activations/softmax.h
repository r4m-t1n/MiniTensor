#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <memory>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "tensors/tensor_math.h"
#include "tensors/tensor_reductions.h"

template<typename T>
std::shared_ptr<Tensor<T>> softmax(std::shared_ptr<Tensor<T>> tensor, int axis = -1) {

    auto max_val_tensor = max(tensor, axis);
    auto exp_tensor = tensor_exp<T, T>(tensor_sub(tensor, max_val_tensor));
    auto sum_exp_tensor = sum(exp_tensor, axis);
    auto result = tensor_div(exp_tensor, sum_exp_tensor);

    return result;
}

#endif