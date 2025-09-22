#ifndef AUTOGRAD_ACTIVATIONS_H
#define AUTOGRAD_ACTIVATIONS_H

#include <vector>
#include <memory>
#include <cmath>
#include "tensors/tensor.h"

template<typename T>
struct ReluBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;

    ReluBackward(std::shared_ptr<Tensor<T>> a) : parent_input(a) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            auto grad_a_data = std::vector<T>(grad_out->size);
            for (int i = 0; i < grad_out->size; ++i) {
                grad_a_data[i] = (parent_input->data[i] > 0) ? grad_out->data[i] : static_cast<T>(0);
            }
            
            auto grad_a = std::make_shared<Tensor<T>>(grad_a_data, grad_out->shape);

            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->size; ++i) {
                    parent_input->grad->data[i] += grad_a->data[i];
                }
            }
            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
};

template<typename T>
struct TanhBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;
    std::shared_ptr<Tensor<T>> parent_output;

    TanhBackward(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) : parent_input(a), parent_output(b) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            auto grad_a_data = std::vector<T>(grad_out->size);
            for (int i = 0; i < grad_out->size; ++i) {
                grad_a_data[i] = grad_out->data[i] * (static_cast<T>(1) - (parent_output->data[i] * parent_output->data[i]));
            }
            
            auto grad_a = std::make_shared<Tensor<T>>(grad_a_data, grad_out->shape);

            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->size; ++i) {
                    parent_input->grad->data[i] += grad_a->data[i];
                }
            }
            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
};

template<typename T>
struct SigmoidBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_output;

    SigmoidBackward(std::shared_ptr<Tensor<T>> output) : parent_output(output) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        auto parent_input = parent_output->parents[0];
        if (parent_input->requires_grad) {
            auto grad_a_data = std::vector<T>(grad_out->size);
            for (int i = 0; i < grad_out->size; ++i) {
                T sig_out = parent_output->data[i];
                grad_a_data[i] = grad_out->data[i] * (sig_out * (static_cast<T>(1) - sig_out));
            }
            
            auto grad_a = std::make_shared<Tensor<T>>(grad_a_data, grad_out->shape);

            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->size; ++i) {
                    parent_input->grad->data[i] += grad_a->data[i];
                }
            }
            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
};

#endif