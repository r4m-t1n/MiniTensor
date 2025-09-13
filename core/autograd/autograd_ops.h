#ifndef AUTOGRAD_OPS_H
#define AUTOGRAD_OPS_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cmath>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "tensors/tensor_broadcast.h"

template<typename T>
struct AddBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> a_parent, b_parent;
    std::vector<int> a_shape, b_shape;

    AddBackward(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
        : a_parent(a), b_parent(b), a_shape(a->shape), b_shape(b->shape) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (a_parent->requires_grad) {
            auto grad_a = std::make_shared<Tensor<T>>(unbroadcast(*grad_out, a_shape));
            if (!a_parent->grad) {
                a_parent->grad = grad_a;
            } else {
                for (int i = 0; i < a_parent->grad->size; ++i) a_parent->grad->data[i] += grad_a->data[i];
            }
            if (a_parent->grad_fn) a_parent->grad_fn->backward(grad_a);
        }
        if (b_parent->requires_grad) {
            auto grad_b = std::make_shared<Tensor<T>>(unbroadcast(*grad_out, b_shape));
            if (!b_parent->grad) {
                b_parent->grad = grad_b;
            } else {
                for (int i = 0; i < b_parent->grad->size; ++i) b_parent->grad->data[i] += grad_b->data[i];
            }
            if (b_parent->grad_fn) b_parent->grad_fn->backward(grad_b);
        }
    }
};

template<typename T>
struct SubBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> a_parent, b_parent;
    std::vector<int> a_shape, b_shape;

    SubBackward(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
        : a_parent(a), b_parent(b), a_shape(a->shape), b_shape(b->shape) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (a_parent->requires_grad) {
            auto grad_a = std::make_shared<Tensor<T>>(unbroadcast(*grad_out, a_shape));
            if (!a_parent->grad) {
                a_parent->grad = grad_a;
            } else {
                for (int i = 0; i < a_parent->grad->size; ++i) a_parent->grad->data[i] += grad_a->data[i];
            }
            if (a_parent->grad_fn) a_parent->grad_fn->backward(grad_a);
        }
        if (b_parent->requires_grad) {
            auto neg_grad = tensor_scalar_mul(grad_out, static_cast<T>(-1));
            auto grad_b = std::make_shared<Tensor<T>>(unbroadcast(*neg_grad, b_shape));
            if (!b_parent->grad) {
                b_parent->grad = grad_b;
            } else {
                for (int i = 0; i < b_parent->grad->size; ++i) b_parent->grad->data[i] += grad_b->data[i];
            }
            if (b_parent->grad_fn) b_parent->grad_fn->backward(neg_grad);
        }
    }
};

template<typename T>
struct MulBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> a_parent, b_parent;
    std::vector<int> a_shape, b_shape;

    MulBackward(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
        : a_parent(a), b_parent(b), a_shape(a->shape), b_shape(b->shape) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (a_parent->requires_grad) {
            auto grad_a_unsummed = tensor_mul(grad_out, b_parent);
            auto grad_a = std::make_shared<Tensor<T>>(unbroadcast(*grad_a_unsummed, a_shape));
            if (!a_parent->grad) {
                a_parent->grad = grad_a;
            } else {
                for (int i = 0; i < a_parent->grad->size; ++i) a_parent->grad->data[i] += grad_a->data[i];
            }
            if (a_parent->grad_fn) a_parent->grad_fn->backward(grad_a);
        }
        if (b_parent->requires_grad) {
            auto grad_b_unsummed = tensor_mul(grad_out, a_parent);
            auto grad_b = std::make_shared<Tensor<T>>(unbroadcast(*grad_b_unsummed, b_shape));
            if (!b_parent->grad) {
                b_parent->grad = grad_b;
            } else {
                for (int i = 0; i < b_parent->grad->size; ++i) b_parent->grad->data[i] += grad_b->data[i];
            }
            if (b_parent->grad_fn) b_parent->grad_fn->backward(grad_b);
        }
    }
};

template<typename T>
struct DivBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> a_parent, b_parent;
    std::vector<int> a_shape, b_shape;

    DivBackward(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
        : a_parent(a), b_parent(b), a_shape(a->shape), b_shape(b->shape) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (a_parent->requires_grad) {
            auto grad_a_unsummed = tensor_div(grad_out, b_parent);
            auto grad_a = std::make_shared<Tensor<T>>(unbroadcast(*grad_a_unsummed, a_shape));
            if (!a_parent->grad) {
                a_parent->grad = grad_a;
            } else {
                for (int i = 0; i < a_parent->grad->size; ++i) a_parent->grad->data[i] += grad_a->data[i];
            }
            if (a_parent->grad_fn) a_parent->grad_fn->backward(grad_a);
        }
        if (b_parent->requires_grad) {
            auto term1 = scalar_tensor_sub(static_cast<T>(0), a_parent);
            auto term2 = tensor_mul(b_parent, b_parent);
            auto term3 = tensor_div(term1, term2);
            auto grad_b_unsummed = tensor_mul(grad_out, term3);
            auto grad_b = std::make_shared<Tensor<T>>(unbroadcast(*grad_b_unsummed, b_shape));
            if (!b_parent->grad) {
                b_parent->grad = grad_b;
            } else {
                for (int i = 0; i < b_parent->grad->size; ++i) b_parent->grad->data[i] += grad_b->data[i];
            }
            if (b_parent->grad_fn) b_parent->grad_fn->backward(grad_b);
        }
    }
};

template<typename T>
struct AddScalarBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    AddScalarBackward(std::shared_ptr<Tensor<T>> a) : parent(a) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            if (!parent->grad) {
                parent->grad = grad_out;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += grad_out->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_out);
        }
    }
};

template<typename T>
struct SubScalarBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    SubScalarBackward(std::shared_ptr<Tensor<T>> a) : parent(a) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            if (!parent->grad) {
                parent->grad = grad_out;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += grad_out->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_out);
        }
    }
};

template<typename T, typename ScalarType>
struct MulScalarBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    ScalarType scalar_val;
    MulScalarBackward(std::shared_ptr<Tensor<T>> a, ScalarType scalar) : parent(a), scalar_val(scalar) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            auto grad_a = tensor_scalar_mul(grad_out, scalar_val);
            if (!parent->grad) {
                parent->grad = grad_a;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += grad_a->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_a);
        }
    }
};

template<typename T, typename ScalarType>
struct DivScalarBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    ScalarType scalar_val;
    DivScalarBackward(std::shared_ptr<Tensor<T>> a, ScalarType scalar) : parent(a), scalar_val(scalar) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            auto grad_a = tensor_scalar_div(grad_out, scalar_val);
            if (!parent->grad) {
                parent->grad = grad_a;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += grad_a->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_a);
        }
    }
};

template<typename T, typename ScalarType>
struct ScalarTensorSubBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    ScalarTensorSubBackward(std::shared_ptr<Tensor<T>> a) : parent(a) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            auto neg_grad = tensor_scalar_mul(grad_out, static_cast<T>(-1));
            if (!parent->grad) {
                parent->grad = neg_grad;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += neg_grad->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(neg_grad);
        }
    }
};

template<typename T, typename ScalarType>
struct ScalarTensorDivBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    ScalarType scalar_val;
    ScalarTensorDivBackward(ScalarType scalar, std::shared_ptr<Tensor<T>> a) : scalar_val(scalar), parent(a) {}
    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            auto term1 = tensor_scalar_mul(parent, static_cast<T>(-1) * scalar_val);
            auto term2 = tensor_mul(parent, parent);
            auto term3 = tensor_div(term1, term2);
            auto grad_a = tensor_mul(grad_out, term3);
            if (!parent->grad) {
                parent->grad = grad_a;
            } else {
                for (int i = 0; i < parent->size; ++i) parent->grad->data[i] += grad_a->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_a);
        }
    }
};

template<typename T>
struct MatMulBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> a, b;
    MatMulBackward(std::shared_ptr<Tensor<T>> input_a, std::shared_ptr<Tensor<T>> input_b) : a(input_a), b(input_b) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (a->requires_grad) {
            auto b_transposed = transpose(b);
            auto grad_a = mat_mul(grad_out, b_transposed);
            if (!a->grad) {
                a->grad = grad_a;
            } else {
                 for (int i = 0; i < a->grad->size; ++i) a->grad->data[i] += grad_a->data[i];
            }
            if (a->grad_fn) a->grad_fn->backward(grad_a);
        }
        if (b->requires_grad) {
            auto a_transposed = transpose(a);
            auto grad_b = mat_mul(a_transposed, grad_out);
            if (!b->grad) {
                b->grad = grad_b;
            } else {
                for (int i = 0; i < b->grad->size; ++i) b->grad->data[i] += grad_b->data[i];
            }
            if (b->grad_fn) b->grad_fn->backward(grad_b);
        }
    }
};

template<typename T>
struct TransposeBackward : public Function<T> {
    std::shared_ptr<Tensor<T>> parent;
    TransposeBackward(std::shared_ptr<Tensor<T>> p) : parent(p) {}

    void backward(std::shared_ptr<Tensor<T>> grad_out) override {
        if (parent->requires_grad) {
            auto grad_in = transpose(grad_out);
            if (!parent->grad) {
                parent->grad = grad_in;
            } else {
                for(int i = 0; i < parent->grad->size; ++i) parent->grad->data[i] += grad_in->data[i];
            }
            if (parent->grad_fn) parent->grad_fn->backward(grad_in);
        }
    }
};

#endif