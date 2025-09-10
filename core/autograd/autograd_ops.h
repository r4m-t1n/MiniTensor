#ifndef AUTOGRAD_OPS_H
#define AUTOGRAD_OPS_H

#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cmath>

template<typename T>
Tensor<T> unbroadcast(const Tensor<T>& grad, const std::vector<int>& target_shape) {
    if (grad.shape == target_shape) {
        return Tensor<T>(grad.data, grad.shape);
    }

    Tensor<T> final_grad(grad.data, grad.shape);
    while (final_grad.ndim > target_shape.size()) {
        final_grad = sum(final_grad, 0);
    }

    for (int i = 0; i < final_grad.ndim; ++i) {
        if (final_grad.shape[i] > target_shape[i]) {
            final_grad = sum(final_grad, i);
        }
    }

    return final_grad;
}

template<typename T>
struct AddBackward : public Function<T> {
    AddBackward(Tensor<T>* a, Tensor<T>* b) {
        parents.push_back(a);
        parents.push_back(b);
    }
    void backward(const Tensor<T>& grad_out) override {
        if (parents[0]->requires_grad) {
            Tensor<T> grad_a = unbroadcast(grad_out, parents[0]->shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(parents[0]->grad) = *(parents[0]->grad) + grad_a;
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            Tensor<T> grad_b = unbroadcast(grad_out, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                *(parents[1]->grad) = *(parents[1]->grad) + grad_b;
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_b);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct SubBackward : public Function<T> {
    SubBackward(Tensor<T>* a, Tensor<T>* b) {
        parents.push_back(a);
        parents.push_back(b);
    }
    void backward(const Tensor<T>& grad_out) override {
        if (parents[0]->requires_grad) {
            Tensor<T> grad_a = unbroadcast(grad_out, parents[0]->shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(parents[0]->grad) = *(parents[0]->grad) + grad_a;
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            std::vector<T> neg_grad_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                neg_grad_data[i] = -grad_out.data[i];
            }
            Tensor<T> neg_grad(neg_grad_data, grad_out.shape);
            Tensor<T> grad_b = unbroadcast(neg_grad, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                *(parents[1]->grad) = *(parents[1]->grad) + grad_b;
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(neg_grad);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct MulBackward : public Function<T> {
    MulBackward(Tensor<T>* a, Tensor<T>* b) {
        parents.push_back(a);
        parents.push_back(b);
    }
    void backward(const Tensor<T>& grad_out) override {
        if (parents[0]->requires_grad) {
            Tensor<T> grad_a_unsummed = grad_out * *(parents[1]);
            Tensor<T> grad_a = unbroadcast(grad_a_unsummed, parents[0]->shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(parents[0]->grad) = *(parents[0]->grad) + grad_a;
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            Tensor<T> grad_b_unsummed = grad_out * *(parents[0]);
            Tensor<T> grad_b = unbroadcast(grad_b_unsummed, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                *(parents[1]->grad) = *(parents[1]->grad) + grad_b;
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_b);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct DivBackward : public Function<T> {
    DivBackward(Tensor<T>* a, Tensor<T>* b) {
        parents.push_back(a);
        parents.push_back(b);
    }
    void backward(const Tensor<T>& grad_out) override {
        if (parents[0]->requires_grad) {
            Tensor<T> grad_a_unsummed = grad_out / *(parents[1]);
            Tensor<T> grad_a = unbroadcast(grad_a_unsummed, parents[0]->shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(parents[0]->grad) = *(parents[0]->grad) + grad_a;
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            Tensor<T> grad_b_unsummed = grad_out * ((static_cast<T>(0) - *(parents[0])) / (*(parents[1]) * *(parents[1])));
            Tensor<T> grad_b = unbroadcast(grad_b_unsummed, parents[1]->shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                *(parents[1]->grad) = *(parents[1]->grad) + grad_b;
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_b);
            }
        }
    }
    std::vector<Tensor<T>*> parents;
};

template<typename T>
struct AddScalarBackward : public Function<T> {
    Tensor<T>* parent;
    AddScalarBackward(Tensor<T>* a) : parent(a) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(grad_out.data, grad_out.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += grad_out.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(grad_out);
            }
        }
    }
};

template<typename T>
struct SubScalarBackward : public Function<T> {
    Tensor<T>* parent;
    SubScalarBackward(Tensor<T>* a) : parent(a) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(grad_out.data, grad_out.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += grad_out.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(grad_out);
            }
        }
    }
};

template<typename T, typename ScalarType>
struct MulScalarBackward : public Function<T> {
    Tensor<T>* parent;
    ScalarType scalar_val;
    MulScalarBackward(Tensor<T>* a, ScalarType scalar) : parent(a), scalar_val(scalar) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                grad_a_data[i] = grad_out.data[i] * static_cast<T>(scalar_val);
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += grad_a.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(grad_a);
            }
        }
    }
};

template<typename T, typename ScalarType>
struct DivScalarBackward : public Function<T> {
    Tensor<T>* parent;
    ScalarType scalar_val;
    DivScalarBackward(Tensor<T>* a, ScalarType scalar) : parent(a), scalar_val(scalar) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                if (scalar_val == static_cast<T>(0)) {
                    throw std::runtime_error("ERROR: Division by zero is not allowed");
                }
                grad_a_data[i] = grad_out.data[i] / static_cast<T>(scalar_val);
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += grad_a.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(grad_a);
            }
        }
    }
};

template<typename T, typename ScalarType>
struct ScalarTensorSubBackward : public Function<T> {
    ScalarType scalar_val;
    Tensor<T>* parent;
    ScalarTensorSubBackward(ScalarType scalar, Tensor<T>* a) : scalar_val(scalar), parent(a) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            std::vector<T> neg_grad_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                neg_grad_data[i] = -grad_out.data[i];
            }
            Tensor<T> neg_grad(neg_grad_data, grad_out.shape);
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(neg_grad.data, neg_grad.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += neg_grad.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(neg_grad);
            }
        }
    }
};

template<typename T, typename ScalarType>
struct ScalarTensorDivBackward : public Function<T> {
    ScalarType scalar_val;
    Tensor<T>* parent;
    ScalarTensorDivBackward(ScalarType scalar, Tensor<T>* a) : scalar_val(scalar), parent(a) {}
    void backward(const Tensor<T>& grad_out) override {
        if (parent->requires_grad) {
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                if (parent->data[i] == static_cast<T>(0)) {
                    throw std::runtime_error("ERROR: Division by zero is not allowed");
                }
                grad_a_data[i] = grad_out.data[i] * (-static_cast<T>(scalar_val) / (parent->data[i] * parent->data[i]));
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parent->grad == nullptr) {
                parent->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parent->size; ++i) {
                    parent->grad->data[i] += grad_a.data[i];
                }
            }
            if (parent->grad_fn) {
                parent->grad_fn->backward(grad_a);
            }
        }
    }
};

template<typename T>
struct MatMulBackward : public Function<T> {
    Tensor<T>* a;
    Tensor<T>* b;

    MatMulBackward(Tensor<T>* input_a, Tensor<T>* input_b) : a(input_a), b(input_b) {}

    void backward(const Tensor<T>& grad_out) override {
        if (a->requires_grad) {
            Tensor<T> b_transposed = transpose(*b);
            Tensor<T> grad_a = mat_mul(grad_out, b_transposed);
            if (a->grad == nullptr) {
                a->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                *(a->grad) = *(a->grad) + grad_a;
            }
        }

        if (b->requires_grad) {
            Tensor<T> a_transposed = transpose(*a);
            Tensor<T> grad_b = mat_mul(a_transposed, grad_out);
            if (b->grad == nullptr) {
                b->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                *(b->grad) = *(b->grad) + grad_b;
            }
        }
    }
};

#endif