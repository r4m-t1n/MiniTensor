#ifndef CONSTANT_VAL_H
#define CONSTANT_VAL_H

#include <vector>
#include "tensors/tensor.h"

template<typename T>
class Constant_Val {
private:
    T value;
public:
    Constant_Val() : value(T{}) {}

    Constant_Val(T val) : value(val) {}

    void initialize(Tensor<T>& weights) {
        if (!weights.data) return;
        for (int i = 0; i < weights.size; ++i) {
            weights.data[i] = value;
        }
    }
};

#endif