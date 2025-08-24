#ifndef INITIALIZER_BASE_H
#define INITIALIZER_BASE_H

#include "tensors/tensor.h"

template<typename T>
class Initializer {
public:
    virtual ~Initializer() = default;
    virtual void initialize(Tensor<T>& weights) = 0;
};

#endif