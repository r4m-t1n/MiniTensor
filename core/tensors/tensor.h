#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <type_traits>
#include <memory>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template<typename T>
class Tensor;

template<typename T>
struct Function {
    virtual void backward(std::shared_ptr<Tensor<T>> grad) = 0;
    virtual ~Function() = default;
};

template<typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    std::unique_ptr<T[]> data;
    std::vector<int> shape;
    int ndim;
    int size;
    std::vector<int> stride;
    bool requires_grad;

    std::shared_ptr<Tensor<T>> grad;
    std::vector<std::shared_ptr<Tensor<T>>> parents;
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

    std::shared_ptr<Tensor<T>> reshape(const std::vector<int>& new_shape) {
        int new_size = 1;
        for(auto dim : new_shape) new_size *= dim;
        if(new_size != this->size) {
            throw std::runtime_error("ERROR: Reshape size mismatch.");
        }
        auto result = std::make_shared<Tensor<T>>(new_shape, this->requires_grad);
        std::copy(this->data.get(), this->data.get() + this->size, result->data.get());
        return result;
    }

    Tensor(const std::vector<int>& shape, bool req_grad = false)
        : shape(shape), ndim(shape.size()), requires_grad(req_grad), grad(nullptr) {
        if (ndim < 1) throw std::invalid_argument("ERROR: Invalid shape.");
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) throw std::invalid_argument("ERROR: Dimension must be positive.");
        stride = compute_stride(shape, ndim);
        data = std::make_unique<T[]>(size);
    }

    Tensor(const std::vector<T>& data_vec, const std::vector<int>& shape, bool req_grad = false)
        : shape(shape), ndim(shape.size()), requires_grad(req_grad), grad(nullptr) {
        if (ndim < 1) throw std::invalid_argument("ERROR: Invalid shape.");
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (size <= 0) throw std::invalid_argument("ERROR: Dimension must be positive.");
        if (data_vec.size() != static_cast<size_t>(size)) throw std::invalid_argument("ERROR: Data size does not match shape size.");
        stride = compute_stride(shape, ndim);
        data = std::make_unique<T[]>(size);
        std::copy(data_vec.begin(), data_vec.end(), data.get());
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    Tensor(Tensor&& other) noexcept
        : data(std::move(other.data)),
          shape(std::move(other.shape)),
          ndim(other.ndim),
          size(other.size),
          stride(std::move(other.stride)),
          requires_grad(other.requires_grad),
          grad(std::move(other.grad)),
          parents(std::move(other.parents)),
          grad_fn(std::move(other.grad_fn)) {
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            shape = std::move(other.shape);
            ndim = other.ndim;
            size = other.size;
            stride = std::move(other.stride);
            requires_grad = other.requires_grad;
            grad = std::move(other.grad);
            parents = std::move(other.parents);
            grad_fn = std::move(other.grad_fn);
        }
        return *this;
    }
    
    void set_data(const Tensor<T>& other) {
        if (this->size != other.size) {
            throw std::runtime_error("ERROR: set_data requires tensors of the same size.");
        }
        std::copy(other.data.get(), other.data.get() + other.size, this->data.get());
    }

    void backward() {
        if (!requires_grad) {
            return;
        }
        if (grad == nullptr) {
            std::vector<T> ones_data(size, static_cast<T>(1));
            grad = std::make_shared<Tensor<T>>(ones_data, shape, false);
        }
        if (grad_fn) {
            grad_fn->backward(grad);
        }
    }

    void zero_grad() {
        if (grad != nullptr) {
            std::fill(grad->data.get(), grad->data.get() + grad->size, static_cast<T>(0));
        }
    }
};

template<typename T>
std::vector<T> to_vector(const Tensor<T>& tensor) {
    std::vector<T> data_vec(tensor.size);
    std::copy(tensor.data.get(), tensor.data.get() + tensor.size, data_vec.begin());
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
        if (i < t.shape.size() - 1) shape_str += ", ";
    }
    shape_str += ")";

    std::string dtype_name;
    if (std::is_same<T, int>::value) dtype_name = "int32";
    else if (std::is_same<T, float>::value) dtype_name = "float32";
    else if (std::is_same<T, double>::value) dtype_name = "float64";
    else dtype_name = "unknown";

    return "<Tensor dtype=" + dtype_name + " shape=" + shape_str + ">";
}
#endif