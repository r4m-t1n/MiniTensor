#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <type_traits>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template<typename T>
class Tensor;

template<typename T>
struct Function {
    virtual void backward(const Tensor<T>& grad) = 0;
    virtual ~Function() = default;
};

template<typename T>
class Tensor {
public:
    T* data;
    std::vector<int> shape;
    int ndim;
    int size;
    std::vector<int> stride;
    bool requires_grad;
    Tensor<T>* grad;
    std::vector<Tensor<T>*> parents;
    std::unique_ptr<Function<T>> grad_fn;

    static std::vector<int> compute_stride(const std::vector<int>& shape, const int ndim) {
        std::vector<int> stride(ndim);
        int acc = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            stride[i] = acc;
            acc *= shape[i];
        }
        return stride;
    }

    Tensor(const std::vector<int>& shape, bool requires_grad = false)
        : shape(shape), ndim(shape.size()), data(nullptr), requires_grad(requires_grad), grad(nullptr) {
        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) {
            throw std::invalid_argument("ERROR: Dimension must be positive.");
        }

        stride = compute_stride(shape, ndim);

        data = new T[size];
    }

    Tensor(const std::vector<T>& data_vec, const std::vector<int>& shape, bool requires_grad = false)
        : shape(shape), ndim(shape.size()), data(nullptr), requires_grad(requires_grad), grad(nullptr) {
        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }

        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) {
            throw std::invalid_argument("ERROR: Dimension must be positive.");
        }

        if (data_vec.size() != static_cast<size_t>(size)) {
            throw std::invalid_argument("ERROR: Data size does not match shape size.");
        }

        stride = compute_stride(shape, ndim);

        data = new T[size];
        if (!data) {
            throw std::bad_alloc();
        }

        for (int i = 0; i < size; ++i) {
            data[i] = data_vec[i];
        }
    }
    
    Tensor() = delete;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    Tensor(Tensor&& other) noexcept
        : data(other.data),
        shape(std::move(other.shape)),
        ndim(other.ndim),
        size(other.size),
        stride(std::move(other.stride)),
        requires_grad(other.requires_grad),
        grad(other.grad),
        parents(std::move(other.parents)),
        grad_fn(std::move(other.grad_fn)) {
        other.data = nullptr;
        other.grad = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data;
            delete grad;
            data = other.data;
            shape = std::move(other.shape);
            ndim = other.ndim;
            size = other.size;
            stride = std::move(other.stride);
            requires_grad = other.requires_grad;
            grad = other.grad;
            parents = std::move(other.parents);
            grad_fn = std::move(other.grad_fn);
            other.data = nullptr;
            other.grad = nullptr;
        }
        return *this;
    }

    Tensor(T* data_ptr, const std::vector<int>& shape, bool requires_grad = false)
        : shape(shape), ndim(shape.size()), requires_grad(requires_grad), grad(nullptr) {
        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) {
            throw std::invalid_argument("ERROR: Dimension must be positive.");
        }
        stride = compute_stride(shape, ndim);

        this->data = new T[size];
        for(int i = 0; i < size; ++i) {
            this->data[i] = data_ptr[i];
        }
    }
    
    ~Tensor() {
        delete[] data;
        delete grad;
    }

    void backward() {
        if (!requires_grad) {
            return;
        }
        if (grad == nullptr) {
            std::vector<T> ones_data(size, static_cast<T>(1));
            grad = new Tensor<T>(ones_data, shape, false);
        }
        if (grad_fn) {
            grad_fn->backward(*grad);
        }
    }

    void zero_grad() {
        if (grad != nullptr) {
            for (int i = 0; i < size; ++i) {
                grad->data[i] = static_cast<T>(0);
            }
        }
    }
};

template<typename T>
std::vector<T> to_vector(const Tensor<T>& tensor) {
    std::vector<T> data_vec(tensor.size);
    for (int i = 0; i < tensor.size; ++i) {
        data_vec[i] = tensor.data[i];
    }
    return data_vec;
}

template<typename T>
pybind11::list to_nested(const Tensor<T>& tensor, int dim=0, int offset=0) {
    pybind11::list nested_list;
    if (dim == tensor.ndim - 1) {
        for (int i = 0; i < tensor.shape[dim]; ++i)
            nested_list.append(tensor.data[offset + i]);
    } else {
        int step = 1;
        for (int i = dim+1; i < tensor.ndim; ++i) step *= tensor.shape[i];
        for (int i = 0; i < tensor.shape[dim]; ++i)
            nested_list.append(to_nested(tensor, dim+1, offset + i*step));
    }
    return nested_list;
}

template<typename T>
pybind11::list to_nested_wrapper(const Tensor<T>& tensor) {
    return to_nested(tensor, 0, 0);
}

template<typename T>
std::string tensor_repr(const Tensor<T>& t) {
    std::string shape_str = "(";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        shape_str += std::to_string(t.shape[i]);
        if (i < t.shape.size() - 1) {
            shape_str += ", ";
        }
    }
    shape_str += ")";

    std::string dtype_name;
    if (std::is_same<T, int>::value) {
        dtype_name = "int32";
    } else if (std::is_same<T, float>::value) {
        dtype_name = "float32";
    } else if (std::is_same<T, double>::value) {
        dtype_name = "float64";
    } else {
        dtype_name = "unknown";
    }

    return "<Tensor dtype=" + dtype_name + " shape=" + shape_str + ">";
}

#endif