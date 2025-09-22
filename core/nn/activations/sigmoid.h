#ifndef SIGMOID_H
#define SIGMOID_H

#include <cmath>
#include <memory>
#include "tensors/tensor.h"
#include "autograd/autograd_activations.h"

template<typename T>
std::shared_ptr<Tensor<T>> sigmoid(std::shared_ptr<Tensor<T>> tensor) {
    auto result = std::make_shared<Tensor<T>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < result->size; ++i) {
        result->data[i] = 1 / (1 + exp(-tensor->data[i]));
    }

    if (tensor->requires_grad) {
        result->parents.push_back(tensor);
        result->grad_fn = std::make_unique<SigmoidBackward<T>>(tensor);
    }
    return result;
}

#endif