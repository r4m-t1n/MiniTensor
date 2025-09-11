#ifndef TANH_H
#define TANH_H

#include <cmath>
#include <memory>
#include <vector>
#include "tensors/tensor.h"
#include "autograd/autograd_activations.h"

template<typename T>
std::shared_ptr<Tensor<T>> tanh_fn(std::shared_ptr<Tensor<T>> tensor) {
    auto result = std::make_shared<Tensor<T>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < result->size; ++i) {
        result->data[i] = std::tanh(tensor->data[i]);
    }

    if (tensor->requires_grad) {
        result->parents.push_back(tensor);
        result->grad_fn = std::make_unique<TanhBackward<T>>(tensor, result);
    }
    return result;
}

#endif