#ifndef MSE_H
#define MSE_H

#include <iostream>
#include "tensors/tensor.h"

template<typename T>
T mse_loss(const Tensor<T>& y, const Tensor<T>& y_hat) {
    check_tensor_validity(y, y_hat);
    T loss = static_cast<T>(0);
    for (int i = 0; i < y.size; ++i) {
        T diff = y.data[i] - y_hat.data[i];
        loss += diff * diff;
    }
    return loss / static_cast<T>(y.size);
}

#endif