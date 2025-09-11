#ifndef MSE_H
#define MSE_H

#include <memory>
#include <stdexcept>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "autograd/autograd_losses.h"

template<typename T>
std::shared_ptr<Tensor<T>> mse_loss(std::shared_ptr<Tensor<T>> y, std::shared_ptr<Tensor<T>> y_hat) {
    check_tensor_validity(y, y_hat);
    T loss_val = static_cast<T>(0);
    for (int i = 0; i < y->size; ++i) {
        T diff = y_hat->data[i] - y->data[i];
        loss_val += diff * diff;
    }
    loss_val /= static_cast<T>(y->size);

    bool result_requires_grad = y->requires_grad || y_hat->requires_grad;
    
    auto result = std::make_shared<Tensor<T>>(std::vector<T>{loss_val}, std::vector<int>{1}, result_requires_grad);

    if (result->requires_grad) {
        result->parents.push_back(y);
        result->parents.push_back(y_hat);
        result->grad_fn = std::make_unique<MseLossBackward<T>>(y, y_hat);
    }

    return result;
}

#endif