#ifndef AUTOGRAD_REDUCTION_H
#define AUTOGRAD_REDUCTION_H

#include <vector>
#include <memory>
#include "tensors/tensor.h"

template<typename T>
struct SumBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;
    std::vector<int> original_shape;
    int axis;

    SumBackward(std::shared_ptr<Tensor<T>> input, int reduction_axis) 
        : parent_input(input), original_shape(input->shape), axis(reduction_axis) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            auto grad_a = std::make_shared<Tensor<T>>(original_shape, false);

            if (axis == -1) {
                 for (int i = 0; i < grad_a->size; ++i) {
                    grad_a->data[i] = grad_out->data[0];
                }
            } else if (axis == 0) {
                for (int i = 0; i < original_shape[0]; ++i) {
                    for (int j = 0; j < grad_out->size; ++j) {
                        int grad_a_index = i * grad_out->size + j;
                        grad_a->data[grad_a_index] = grad_out->data[j];
                    }
                }
            }
            
            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->grad->size; ++i) {
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
struct MeanBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;
    std::vector<int> original_shape;
    int axis;

    MeanBackward(std::shared_ptr<Tensor<T>> input, int reduction_axis)
        : parent_input(input), original_shape(input->shape), axis(reduction_axis) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            int n_elements;
            if (axis == -1) {
                n_elements = parent_input->size;
            } else {
                n_elements = parent_input->shape[axis];
            }
            
            auto grad_a = std::make_shared<Tensor<T>>(original_shape, false);

            if (axis == -1) {
                T grad_val = grad_out->data[0] / static_cast<T>(n_elements);
                 for (int i = 0; i < grad_a->size; ++i) {
                    grad_a->data[i] = grad_val;
                }
            } else if (axis == 0) {
                for (int i = 0; i < original_shape[0]; ++i) {
                    for (int j = 0; j < grad_out->size; ++j) {
                        int grad_a_index = i * grad_out->size + j;
                        grad_a->data[grad_a_index] = grad_out->data[j] / static_cast<T>(n_elements);
                    }
                }
            }
            
            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->grad->size; ++i) {
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
struct MaxBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;
    std::vector<int> max_indices;

    MaxBackward(std::shared_ptr<Tensor<T>> input, std::vector<int> indices) 
        : parent_input(input), max_indices(indices) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            auto grad_a = std::make_shared<Tensor<T>>(parent_input->shape, false);
            std::fill(grad_a->data.get(), grad_a->data.get() + grad_a->size, static_cast<T>(0));
            
            for(size_t i = 0; i < max_indices.size(); ++i) {
                grad_a->data[max_indices[i]] = grad_out->data[i];
            }

            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->grad->size; ++i) {
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
struct MinBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent_input;
    std::vector<int> min_indices;

    MinBackward(std::shared_ptr<Tensor<T>> input, std::vector<int> indices) 
        : parent_input(input), min_indices(indices) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent_input->requires_grad) {
            auto grad_a = std::make_shared<Tensor<T>>(parent_input->shape, false);
            std::fill(grad_a->data.get(), grad_a->data.get() + grad_a->size, static_cast<T>(0));

            for(size_t i = 0; i < min_indices.size(); ++i) {
                grad_a->data[min_indices[i]] = grad_out->data[i];
            }

            if (!parent_input->grad) {
                parent_input->grad = grad_a;
            } else {
                for (int i = 0; i < parent_input->grad->size; ++i) {
                    parent_input->grad->data[i] += grad_a->data[i];
                }
            }

            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
};

/* to-do:
argmin
argmax
*/

#endif