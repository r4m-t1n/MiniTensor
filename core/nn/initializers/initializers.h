#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "nn/initializers/base.h"
#include <random>
#include <cmath>

template<typename T>
class HeNormal : public Initializer<T> {
public:
    void initialize(Tensor<T>& weights) override {
        if (weights.data == nullptr || weights.ndim < 2) return;

        size_t fan_in = weights.shape[0];
        double std_dev = std::sqrt(2.0 / fan_in);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0, std_dev);

        for (int i = 0; i < weights.size; ++i) {
            weights.data[i] = dist(gen);
        }
    }
};

template<typename T>
class XavierUniform : public Initializer<T> {
public:
    void initialize(Tensor<T>& weights) override {
        if (weights.data == nullptr || weights.ndim < 2) return;

        size_t fan_in = weights.shape[0];
        size_t fan_out = weights.shape[1];
        double limit = std::sqrt(6.0 / (fan_in + fan_out));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-limit, limit);

        for (int i = 0; i < weights.size; ++i) {
            weights.data[i] = dist(gen);
        }
    }
};

template<typename T>
class Constant_Val : public Initializer<T> {
private:
    T value;
public:
    Constant_Val(T val) : value(val) {}

    void initialize(Tensor<T>& weights) override {
        if (weights.data == nullptr) return;
        for (int i = 0; i < weights.size; ++i) {
            weights.data[i] = value;
        }
    }
};

#endif