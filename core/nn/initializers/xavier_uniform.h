#ifndef XAVIER_UNIFORM_H
#define XAVIER_UNIFORM_H

#include <random>
#include <cmath>
#include <cstddef>
#include "tensors/tensor.h"

template<typename T>
class XavierUniform {
public:
    void initialize(Tensor<T>& weights) {
        if (weights.data == nullptr || weights.ndim < 2) return;

        size_t fan_in = weights.shape[1];
        size_t fan_out = weights.shape[0];
        double limit = std::sqrt(6.0 / (fan_in + fan_out));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-limit, limit);

        for (int i = 0; i < weights.size; ++i) {
            weights.data[i] = dist(gen);
        }
    }
};

#endif