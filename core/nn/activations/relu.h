#ifndef RELU_H
#define RELU_H

#include <memory>
#include "tensors/tensor.h"
#include "autograd/autograd_activations.h"

template<typename T>
std::shared_ptr<Tensor<T>> relu(std::shared_ptr<Tensor<T>> tensor) {
    auto result = std::make_shared<Tensor<T>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < result->size; ++i) {
        result->data[i] = (tensor->data[i] > 0) ? tensor->data[i] : static_cast<T>(0);
    }

    if (tensor->requires_grad) {
        result->parents.push_back(tensor);
        result->grad_fn = std::make_unique<ReluBackward<T>>(tensor);
    }
    return result;
}

#endif