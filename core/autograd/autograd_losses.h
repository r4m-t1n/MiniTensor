#ifndef AUTOGRAD_LOSSES_H
#define AUTOGRAD_LOSSES_H

#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include "tensors/tensor.h"

template<typename T>
struct MseLossBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> y_true, y_pred;

    MseLossBackward(std::shared_ptr<Tensor<T>> y, std::shared_ptr<Tensor<T>> y_hat)
        : y_true(y), y_pred(y_hat) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (y_pred->requires_grad) {
            T n_elements = static_cast<T>(y_true->size);
            auto grad_y_hat_data = std::vector<T>(y_pred->size);

            for (int i = 0; i < y_pred->size; ++i) {
                grad_y_hat_data[i] = grad_out->data[0] * static_cast<T>(2) * (y_pred->data[i] - y_true->data[i]) / n_elements;
            }

            auto grad_y_hat = std::make_shared<Tensor<T>>(grad_y_hat_data, y_pred->shape);

            if (!y_pred->grad) {
                y_pred->grad = grad_y_hat;
            } else {
                for (int i = 0; i < y_pred->size; ++i) {
                    y_pred->grad->data[i] += grad_y_hat->data[i];
                }
            }

            if (y_pred->grad_fn) {
                y_pred->grad_fn->backward(grad_y_hat);
            }
        }
    }
};

template<typename T>
struct MaeLossBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> y_true, y_pred;

    MaeLossBackward(std::shared_ptr<Tensor<T>> y, std::shared_ptr<Tensor<T>> y_hat)
        : y_true(y), y_pred(y_hat) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (y_pred->requires_grad) {
            T n_elements = static_cast<T>(y_true->size);
            auto grad_y_hat_data = std::vector<T>(y_pred->size);

            for (int i = 0; i < y_pred->size; ++i) {
                T diff = y_pred->data[i] - y_true->data[i];
                T sign = (diff > 0) ? static_cast<T>(1) : ((diff < 0) ? static_cast<T>(-1) : static_cast<T>(0));
                grad_y_hat_data[i] = grad_out->data[0] * sign / n_elements;
            }

            auto grad_y_hat = std::make_shared<Tensor<T>>(grad_y_hat_data, y_pred->shape);

            if (!y_pred->grad) {
                y_pred->grad = grad_y_hat;
            } else {
                for (int i = 0; i < y_pred->size; ++i) {
                    y_pred->grad->data[i] += grad_y_hat->data[i];
                }
            }

            if (y_pred->grad_fn) {
                y_pred->grad_fn->backward(grad_y_hat);
            }
        }
    }
};

template<typename T>
struct BceLossBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> y_true, y_pred;

    BceLossBackward(std::shared_ptr<Tensor<T>> y, std::shared_ptr<Tensor<T>> y_hat)
        : y_true(y), y_pred(y_hat) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (y_pred->requires_grad) {
            T n_elements = static_cast<T>(y_true->size);
            auto grad_y_hat_data = std::vector<T>(y_pred->size);

            for (int i = 0; i < y_pred->size; ++i) {
                T y = y_true->data[i];
                T y_hat = y_pred->data[i];

                grad_y_hat_data[i] = grad_out->data[0] * ( (y_hat - y) / ( (y_hat * (1 - y_hat)) * n_elements ) );
            }

            auto grad_y_hat = std::make_shared<Tensor<T>>(grad_y_hat_data, y_pred->shape);

            if (!y_pred->grad) {
                y_pred->grad = grad_y_hat;
            } else {
                for (int i = 0; i < y_pred->size; ++i) {
                    y_pred->grad->data[i] += grad_y_hat->data[i];
                }
            }

            if (y_pred->grad_fn) {
                y_pred->grad_fn->backward(grad_y_hat);
            }
        }
    }
};

#endif