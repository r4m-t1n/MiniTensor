#ifndef LINEAR_H
#define LINEAR_H

#include "nn/base.h"
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "nn/initializers/base.h"
#include "nn/initializers/initializers.h" 
#include <vector>
#include <memory>

template<typename T>
class Linear : public Layer<T> {
private:
    Tensor<T> weights;
    Tensor<T> bias;

    std::unique_ptr<Tensor<T>> input_cache;

public:
    Linear(int input_features, int output_features,
           std::unique_ptr<Initializer<T>> weight_init = std::make_unique<HeNormal<T>>(),
           std::unique_ptr<Initializer<T>> bias_init = std::make_unique<Constant_Val<T>>(0.0f))
        : weights({input_features, output_features}, true),
          bias({1, output_features}, true) {
        
        weight_init->initialize(this->weights);
        bias_init->initialize(this->bias);
    }

    Tensor<T> forward(const Tensor<T>& input) override {

        this->input_cache = std::make_unique<Tensor<T>>(input.data, input.shape);


        Tensor<T> output = mat_mul(input, this->weights);
        output = output + this->bias; 

        return output;
    }

    Tensor<T> backward(const Tensor<T>& output_gradient) override {
        Tensor<T> grad_bias = sum(output_gradient, 0);

        Tensor<T> input_transposed = transpose(*(this->input_cache));
        Tensor<T> grad_weights = mat_mul(input_transposed, output_gradient);

        Tensor<T> weights_transposed = transpose(this->weights);
        Tensor<T> grad_input = mat_mul(output_gradient, weights_transposed);

        if (this->bias.grad == nullptr) {
            this->bias.grad = new Tensor<T>(grad_bias.data, grad_bias.shape);
        } else {
            *(this->bias.grad) = *(this->bias.grad) + grad_bias;
        }

        if (this->weights.grad == nullptr) {
            this->weights.grad = new Tensor<T>(grad_weights.data, grad_weights.shape);
        } else {
            *(this->weights.grad) = *(this->weights.grad) + grad_weights;
        }
        
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&weights, &bias};
    }
};

#endif