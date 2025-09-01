#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <iostream>
#include <stdexcept>
#include "tensor.h"
#include <vector>
#include <cmath>

template<typename T_input, typename T_output>
Tensor<T_output> tensor_sqrt(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        if (value < 0) {
            throw std::runtime_error("ERROR: Cannot compute the square root of a negative number.");
        }
        result.data[i] = std::sqrt(value);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_log(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        if (value <= 0) {
            throw std::runtime_error("ERROR: Cannot compute the log of a non-positive number.");
        }
        result.data[i] = std::log(value);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_exp(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        result.data[i] = std::exp(value);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_pow(const Tensor<T_input>& tensor, float exponent) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        result.data[i] = std::pow(value, exponent);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_sin(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        result.data[i] = std::sin(value);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_cos(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        result.data[i] = std::cos(value);
    }

    return result;
}

template<typename T_input, typename T_output>
Tensor<T_output> tensor_tan(const Tensor<T_input>& tensor) {
    Tensor<T_output> result(tensor.shape, tensor.requires_grad);

    for (int i = 0; i < tensor.size; ++i) {
        T_output value = static_cast<T_output>(tensor.data[i]);

        result.data[i] = std::tan(value);
    }

    return result;
}


#endif