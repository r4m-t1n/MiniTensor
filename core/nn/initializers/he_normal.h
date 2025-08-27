#ifndef HE_NORMAL_H
#define HE_NORMAL_H

#include <random>
#include <cmath>

template<typename T>
class HeNormal {
public:
    void initialize(Tensor<T>& weights) {
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

#endif