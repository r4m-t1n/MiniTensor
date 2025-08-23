#ifndef AUTOGRAD_REDUCTION_H
#define AUTOGRAD_REDUCTION_H

#include "tensors/tensor.h"
#include <vector>

template<typename T>
struct SumBackward : public Function<T> {
    Tensor<T>* parent_input;
    std::vector<int> original_shape;
    int axis;

    SumBackward(Tensor<T>* input, int reduction_axis) 
        : parent_input(input), original_shape(input->shape), axis(reduction_axis) {}

    void backward(const Tensor<T>& grad_out) override {
        if (parent_input->requires_grad) {
            Tensor<T> grad_a(original_shape, false);

            if (axis == 0) {
                for (int i = 0; i < original_shape[0]; ++i) {
                    for (int j = 0; j < grad_out.size; ++j) {
                        int grad_a_index = i * grad_out.size + j;
                        grad_a.data[grad_a_index] = grad_out.data[j];
                    }
                }
            }/* else {
               Not implemented for other axes for now 
            }*/
            
            if (parent_input->grad == nullptr) {
                parent_input->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(parent_input->grad) = *(parent_input->grad) + grad_a;
            }

            if (parent_input->grad_fn) {
                parent_input->grad_fn->backward(grad_a);
            }
        }
    }
};

/* to-do:
mean
min
max
argmin
argmax
*/

#endif