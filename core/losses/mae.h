#ifndef MAE_H
#define MAE_H

#include <cmath>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "autograd/autograd_losses.h"

template<typename T>
Tensor<T> mae_loss(Tensor<T>& y, Tensor<T>& y_hat) {
    check_tensor_validity(y, y_hat);
    T loss_val = static_cast<T>(0);
    for (int i = 0; i < y_hat.size; ++i) {
        loss_val += std::abs(y_hat.data[i] - y.data[i]);
    }
    loss_val /= static_cast<T>(y_hat.size);

    bool result_requires_grad = y.requires_grad || y_hat.requires_grad;
    Tensor<T> result({loss_val}, {1}, result_requires_grad);

    if (result.requires_grad) {
        result.parents.push_back(&y);
        result.parents.push_back(&y_hat);
        result.grad_fn = std::make_unique<MaeLossBackward<T>>(&y, &y_hat);
    }

    return result;
}


#endif