#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <iostream>
#include <stdexcept>
#include "tensor.h"
#include <vector>
#include <memory>
#include <cmath>

template<typename T>
void check_tensor_validity(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.ndim != b.ndim) {
        throw std::invalid_argument(
            "ERROR: Shapes are not the same: " +
            std::to_string(a.ndim) + " and " + std::to_string(b.ndim)
        );
    }
    for (int i = 0; i < a.ndim; ++i) {
        if (a.shape[i] != b.shape[i]){
            throw std::invalid_argument(
                "ERROR: Shapes are not the same at dimension " + std::to_string(i) + ": " +
                std::to_string(a.shape[i]) + " and " + std::to_string(b.shape[i])
            );
        }
    }
}

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
Tensor<T> tensor_add(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<AddBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T>
Tensor<T> tensor_sub(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - b.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<SubBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T>
Tensor<T> tensor_mul(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] * b.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<MulBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T>
Tensor<T> tensor_div(const Tensor<T>& a, const Tensor<T>& b) {
    check_tensor_validity(a, b);
    Tensor<T> result(a.shape, a.requires_grad || b.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        if (b.data[i] == static_cast<T>(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = a.data[i] / b.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.parents.push_back(const_cast<Tensor<T>*>(&b));
        result.grad_fn = std::make_unique<DivBackward<T>>(const_cast<Tensor<T>*>(&a), const_cast<Tensor<T>*>(&b));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_add(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<AddScalarBackward<T>>(const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_sub(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<SubScalarBackward<T>>(const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_sub(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = static_cast<T>(scalar) - a.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<ScalarTensorSubBackward<T, ScalarType>>(scalar, const_cast<Tensor<T>*>(&a));
    }
    return result;
}


template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_mul(const Tensor<T>& a, ScalarType scalar) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] * static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<MulScalarBackward<T, ScalarType>>(const_cast<Tensor<T>*>(&a), scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> tensor_scalar_div(const Tensor<T>& a, ScalarType scalar) {
    if (static_cast<T>(scalar) == T(0)) {
        throw std::runtime_error("ERROR: Division by zero is not allowed");
    }
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] / static_cast<T>(scalar);
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<DivScalarBackward<T, ScalarType>>(const_cast<Tensor<T>*>(&a), scalar);
    }
    return result;
}

template<typename T, typename ScalarType>
Tensor<T> scalar_tensor_div(ScalarType scalar, const Tensor<T>& a) {
    Tensor<T> result(a.shape, a.requires_grad);
    for (int i = 0; i < a.size; ++i) {
        if (a.data[i] == T(0)) {
            throw std::runtime_error("ERROR: Division by zero is not allowed");
        }
        result.data[i] = static_cast<T>(scalar) / a.data[i];
    }
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor<T>*>(&a));
        result.grad_fn = std::make_unique<ScalarTensorDivBackward<T, ScalarType>>(scalar, const_cast<Tensor<T>*>(&a));
    }
    return result;
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_add(t1, t2);
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_sub(t1, t2);
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_mul(t1, t2);
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& t1, const Tensor<T>& t2) {
    return tensor_div(t1, t2);
}

template<typename T, typename U>
Tensor<T> operator+(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_add(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator-(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_sub(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator*(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_mul(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator/(const Tensor<T>& tensor, U scalar) {
    return tensor_scalar_div(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator+(U scalar, const Tensor<T>& tensor) {
    return tensor_scalar_add(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator-(U scalar, const Tensor<T>& tensor) {
    return scalar_tensor_sub(scalar, tensor); 
}

template<typename T, typename U>
Tensor<T> operator*(U scalar, const Tensor<T>& tensor) {
    return tensor_scalar_mul(tensor, scalar);
}

template<typename T, typename U>
Tensor<T> operator/(U scalar, const Tensor<T>& tensor) {
    return scalar_tensor_div(scalar, tensor);
}

template<typename T>
Tensor<T> mat_mul(Tensor<T> &a, Tensor<T> &b){
    if (a.ndim != 2 || b.ndim != 2){
        throw std::invalid_argument("ERROR: Both tensors must be 2D matrices");
    }

    int m = a.shape[0];
    int k1 = a.shape[1];
    int k2 = b.shape[0];
    int n = b.shape[1];

    if (k1 != k2){
        throw std::invalid_argument(
            "ERROR: Shapes are not valid to multiply: " +
            std::to_string(m) + "x" + std::to_string(k1) +
            " and " +
            std::to_string(k2) + "x" + std::to_string(n)
        );
    }

    Tensor<T> result({m, n});

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T sum = 0;
            for (int r = 0; r < k1; r++) {
                int idx_a = i * a.stride[0] + r * a.stride[1];
                int idx_b = r * b.stride[0] + j * b.stride[1];
                sum += a.data[idx_a] * b.data[idx_b];
            }
            int idx_res = i * result.stride[0] + j * result.stride[1];
            result.data[idx_res] = sum;
        }
    }

    return result;

}

#endif