#ifndef LOSSES_H
#define LOSSES_H

#include <iostream>
#include <stdexcept>
#include "../tensors/tensor.h"
#include <vector>
#include <memory>
#include <cmath>

template<typename T>
struct MseLossBackward : public Function<T> {
    MseLossBackward(Tensor<T>* y, Tensor<T>* y_hat) {
        parents.push_back(y);
        parents.push_back(y_hat);
    }
    void backward(const Tensor<T>& grad_out) override {
        T n_elements = static_cast<T>(parents[0]->size);
        if (parents[1]->requires_grad) {
            std::vector<T> grad_y_hat_data(parents[1]->size);
            for (int i = 0; i < parents[1]->size; ++i) {
                grad_y_hat_data[i] = static_cast<T>(2) * (parents[1]->data[i] - parents[0]->data[i]) / n_elements;
            }
            Tensor<T> grad_y_hat(grad_y_hat_data, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_y_hat.data, grad_y_hat.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += grad_y_hat.data[i];
                }
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_y_hat);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct MaeLossBackward : public Function<T> {
    MaeLossBackward(Tensor<T>* y, Tensor<T>* y_hat) {
        parents.push_back(y);
        parents.push_back(y_hat);
    }
    void backward(const Tensor<T>& grad_out) override {
        T n_elements = static_cast<T>(parents[0]->size);
        if (parents[1]->requires_grad) {
            std::vector<T> grad_y_hat_data(parents[1]->size);
            for (int i = 0; i < parents[1]->size; ++i) {
                T diff = parents[1]->data[i] - parents[0]->data[i];
                grad_y_hat_data[i] = (diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0)) / n_elements;
            }
            Tensor<T> grad_y_hat(grad_y_hat_data, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_y_hat.data, grad_y_hat.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += grad_y_hat.data[i];
                }
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_y_hat);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

#endif