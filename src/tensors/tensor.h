#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template<typename T>
class Tensor {
public:
    T* data;
    std::vector<int> shape;
    int ndim;
    int size;

    Tensor(const std::vector<int>& shape) 
        : shape(shape), ndim(shape.size()), data(nullptr) {

        if (ndim < 1) {
            throw std::invalid_argument("ERROR: Invalid shape.");
        }

        size = 1;
        for (int dim : shape) {
            if (dim <= 0)
                throw std::invalid_argument("ERROR: Dimension must be positive.");
            size *= dim;
        }

        data = new T[size];

        if (!data) {
            throw std::bad_alloc();
        }
    }

    ~Tensor() {
        delete[] data;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};

#endif