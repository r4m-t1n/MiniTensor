#ifndef LINEAR_H
#define LINEAR_H

#include <memory>
#include <vector>
#include <variant>
#include <type_traits>
#include <string>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "nn/initializers/initializers.h"

template<typename T>
class Linear;

template<typename T>
std::string linear_repr(const Linear<T>& linear_layer);

template<typename T>
using Initializer = typename std::conditional_t<
    std::is_floating_point_v<T>,
    std::variant<std::shared_ptr<HeNormal<T>>, std::shared_ptr<XavierUniform<T>>, std::shared_ptr<Constant_Val<T>>>,
    std::variant<std::shared_ptr<Constant_Val<T>>>
>;

template<typename T>
class Linear {
private:
    std::shared_ptr<Tensor<T>> weights;
    std::shared_ptr<Tensor<T>> bias;
    std::shared_ptr<Tensor<T>> input_cache; 
    
    int input_f;
    int output_f;
    friend std::string linear_repr<T>(const Linear<T>&);

public:
    Linear(int input_features, int output_features,
           Initializer<T> weight_init,
           Initializer<T> bias_init)
        : input_f(input_features), output_f(output_features) {
        
        weights = std::make_shared<Tensor<T>>(std::vector<int>{output_features, input_features}, true);
        bias = std::make_shared<Tensor<T>>(std::vector<int>{1, output_features}, true);
        
        std::visit([this](auto&& arg){ arg->initialize(*(this->weights)); }, weight_init);
        std::visit([this](auto&& arg){ arg->initialize(*(this->bias)); }, bias_init);
    }

    std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input) {
        this->input_cache = input;

        auto transposed_weights = transpose(this->weights); 
        
        auto output = mat_mul(input, transposed_weights);
        return tensor_add(output, this->bias);
    }

    std::vector<std::shared_ptr<Tensor<T>>> parameters() {
        return {weights, bias};
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