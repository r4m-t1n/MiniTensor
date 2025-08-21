#ifndef TANH_H
#define TANH_H

#include "tensors/tensor.h"
#include "autograd/autograd_activations.h"
#include <cmath>

template<typename T>
Tensor<T> tanh(Tensor<T>& tensor) {
    Tensor<T> result(tensor.shape, tensor.requires_grad);
    for (int i = 0; i < result.size; ++i) {
        result.data[i] = std::tanh(tensor.data[i]);
    }
    if (tensor.requires_grad) {
        result.parents.push_back(&tensor);
        result.grad_fn = std::make_unique<TanhBackward<T>>(&tensor, &result);
    }
    return result;
}

#endif