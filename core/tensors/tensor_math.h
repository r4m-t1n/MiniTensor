#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "tensor.h"

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_sqrt(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        T_output value = static_cast<T_output>(tensor->data[i]);
        if (value < 0) {
            throw std::runtime_error("ERROR: Cannot compute the square root of a negative number.");
        }
        result->data[i] = std::sqrt(value);
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_log(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        T_output value = static_cast<T_output>(tensor->data[i]);
        if (value <= 0) {
            throw std::runtime_error("ERROR: Cannot compute the log of a non-positive number.");
        }
        result->data[i] = std::log(value);
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_exp(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        result->data[i] = std::exp(static_cast<T_output>(tensor->data[i]));
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_pow(const std::shared_ptr<Tensor<T_input>>& tensor, float exponent) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        result->data[i] = std::pow(static_cast<T_output>(tensor->data[i]), exponent);
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_sin(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        result->data[i] = std::sin(static_cast<T_output>(tensor->data[i]));
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_cos(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        result->data[i] = std::cos(static_cast<T_output>(tensor->data[i]));
    }
    return result;
}

template<typename T_input, typename T_output>
std::shared_ptr<Tensor<T_output>> tensor_tan(const std::shared_ptr<Tensor<T_input>>& tensor) {
    auto result = std::make_shared<Tensor<T_output>>(tensor->shape, tensor->requires_grad);

    for (int i = 0; i < tensor->size; ++i) {
        result->data[i] = std::tan(static_cast<T_output>(tensor->data[i]));
    }
    return result;
}

#endif