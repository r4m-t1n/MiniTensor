#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include <iostream>
#include <stdexcept>
#include "tensors/tensor.h"
#include <vector>
#include <memory>
#include <cmath>

template<typename T>
struct AddBackward : public Function<T> {
    AddBackward(Tensor<T>* a, Tensor<T>* b) {
        parents.push_back(a);
        parents.push_back(b);
    }
    void backward(const Tensor<T>& grad_out) override {
        if (parents[0]->requires_grad) {
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_out.data, grad_out.shape);
            } else {
                for (int i = 0; i < parents[0]->size; ++i) {
                    parents[0]->grad->data[i] += grad_out.data[i];
                }
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_out);
            }
        }
        if (parents[1]->requires_grad) {
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_out.data, grad_out.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += grad_out.data[i];
                }
            }
            if (parents[1]->grad_fn) {
                parents[1]->grad_fn->backward(grad_out);
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
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_out.data, grad_out.shape);
            } else {
                for (int i = 0; i < parents[0]->size; ++i) {
                    parents[0]->grad->data[i] += grad_out.data[i];
                }
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_out);
            }
        }
        if (parents[1]->requires_grad) {
            std::vector<T> neg_grad_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                neg_grad_data[i] = -grad_out.data[i];
            }
            Tensor<T> neg_grad(neg_grad_data, grad_out.shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(neg_grad.data, neg_grad.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += neg_grad.data[i];
                }
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
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                grad_a_data[i] = grad_out.data[i] * parents[1]->data[i];
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parents[0]->size; ++i) {
                    parents[0]->grad->data[i] += grad_a.data[i];
                }
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            std::vector<T> grad_b_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                grad_b_data[i] = grad_out.data[i] * parents[0]->data[i];
            }
            Tensor<T> grad_b(grad_b_data, grad_out.shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += grad_b.data[i];
                }
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
            std::vector<T> grad_a_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                if (parents[1]->data[i] == static_cast<T>(0)) {
                    throw std::runtime_error("ERROR: Division by zero is not allowed");
                }
                grad_a_data[i] = grad_out.data[i] / parents[1]->data[i];
            }
            Tensor<T> grad_a(grad_a_data, grad_out.shape);
            if (parents[0]->grad == nullptr) {
                parents[0]->grad = new Tensor<T>(grad_a.data, grad_a.shape);
            } else {
                for (int i = 0; i < parents[0]->size; ++i) {
                    parents[0]->grad->data[i] += grad_a.data[i];
                }
            }
            if (parents[0]->grad_fn) {
                parents[0]->grad_fn->backward(grad_a);
            }
        }
        if (parents[1]->requires_grad) {
            std::vector<T> grad_b_data(grad_out.size);
            for (int i = 0; i < grad_out.size; ++i) {
                if (parents[1]->data[i] == static_cast<T>(0)) {
                    throw std::runtime_error("ERROR: Division by zero is not allowed");
                }
                grad_b_data[i] = grad_out.data[i] * (-parents[0]->data[i] / (parents[1]->data[i] * parents[1]->data[i]));
            }
            Tensor<T> grad_b(grad_b_data, grad_out.shape);
            if (parents[1]->grad == nullptr) {
                parents[1]->grad = new Tensor<T>(grad_b.data, grad_b.shape);
            } else {
                for (int i = 0; i < parents[1]->size; ++i) {
                    parents[1]->grad->data[i] += grad_b.data[i];
                }
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