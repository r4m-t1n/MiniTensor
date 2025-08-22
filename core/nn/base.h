#ifndef BASE_H
#define BASE_H

#include "tensors/tensor.h"
#include <vector>

template<typename T>
class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual Tensor<T> backward(const Tensor<T>& output_gradient) = 0;

    virtual std::vector<Tensor<T>*> parameters() {
        return {};
    }

    void train() { is_training = true; }
    void eval() { is_training = false; }

protected:
    bool is_training = true;
};

#endif