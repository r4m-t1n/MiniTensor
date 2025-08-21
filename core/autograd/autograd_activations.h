#ifndef AUTOGRAD_ACTIVATIONS_H
#define AUTOGRAD_ACTIVATIONS_H

#include <vector>
#include <memory>
#include <cmath>
#include "tensors/tensor.h"

template<typename T>
struct ReluBackward : public Function<T> {
    Tensor<T>* parent_input;

    ReluBackward(Tensor<T>* a) : parent_input(a) {}

    void backward(const Tensor<T>& grad_out) override {
        if (parent_input->requires_grad) {
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                grad_a_data[i] = (parent_input->data[i] > 0) ? grad_out.data[i] : static_cast<T>(0);
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parent_input->grad == nullptr) {
                parent_input->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parent_input->size; ++i) {
                    parent_input->grad->data[i] += grad_a.data[i];
                }
            }
            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct TanhBackward : public Function<T> {
    Tensor<T>* parent_input;
    Tensor<T>* parent_output;

    TanhBackward(Tensor<T>* a, Tensor<T>* b) : parent_input(a), parent_output(b) {}

    void backward(const Tensor<T>& grad_out) override {
        if (parent_input->requires_grad) {
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                grad_a_data[i] = grad_out.data[i] * (1.0 - (parent_output->data[i] * parent_output->data[i]));
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parent_input->grad == nullptr) {
                parent_input->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parent_input->size; ++i) {
                    parent_input->grad->data[i] += grad_a.data[i];
                }
            }
            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

#endif