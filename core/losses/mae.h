#ifndef MAE_H
#define MAE_H

#include <cmath>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"

template<typename T>
T mae_loss(const Tensor<T>& y, const Tensor<T>& y_hat) {
    check_tensor_validity(y, y_hat);
    T loss = static_cast<T>(0);
    for (int i = 0; i < y_hat.size; ++i) {
        loss += std::abs(y_hat.data[i] - y.data[i]);
    }

    return loss / static_cast<T>(y_hat.size);
}

#endif