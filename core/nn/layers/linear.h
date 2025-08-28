#ifndef LINEAR_H
#define LINEAR_H

#include <variant>
#include <type_traits>
#include <string>
#include "tensors/tensor.h"
#include "nn/initializers/initializers.h"

template<typename T>
using Initializer = typename std::conditional_t<
    std::is_floating_point_v<T>,
    std::variant<HeNormal<T>, XavierUniform<T>, Constant_Val<T>>,
    std::variant<Constant_Val<T>>
>;

template<typename T>
class Linear {
private:
    Tensor<T> weights;
    Tensor<T> bias;
    std::unique_ptr<Tensor<T>> input_cache;
    int input_f;
    int output_f;
    friend std::string linear_repr<T>(const Linear<T>&);

public:
    Linear(int input_features, int output_features,
           Initializer<T> weight_init,
           Initializer<T> bias_init)
        : weights({input_features, output_features}, true),
          bias({1, output_features}, true),
          input_f(input_features),
          output_f(output_features) {
        
        std::visit([this](auto&& arg){ arg.initialize(this->weights); }, weight_init);
        std::visit([this](auto&& arg){ arg.initialize(this->bias); }, bias_init);
    }

    Tensor<T> forward(const Tensor<T>& input) {
        this->input_cache = std::make_unique<Tensor<T>>(input.data, input.shape);
        Tensor<T> output = mat_mul(input, this->weights);
        return output + this->bias;
    }

    Tensor<T> backward(const Tensor<T>& output_gradient) {
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

    std::vector<Tensor<T>*> parameters() {
        return {&weights, &bias};
    }
};

template<typename T>
std::string linear_repr(const Linear<T>& linear_layer) {
    std::string dtype_name;
    if (std::is_same_v<T, int>) {
        dtype_name = "int32";
    } else if (std::is_same_v<T, float>) {
        dtype_name = "float32";
    } else if (std::is_same_v<T, double>) {
        dtype_name = "float64";
    } else {
        dtype_name = "unknown";
    }

    return "Linear(in_features=" + std::to_string(linear_layer.input_f) +
           ", out_features=" + std::to_string(linear_layer.output_f) +
           ", dtype='" + dtype_name + "')";
}


#endif