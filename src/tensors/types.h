#ifndef TYPES_H
#define TYPES_H

enum DType{DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE};

struct Tensor{
    void* data;
    int* shape;
    int ndim;
    int size;
    enum DType dtype;
};

struct Tensor* create_tensor(int* shape, int ndim, enum DType dtype);

void free_tensor(struct Tensor* tensor);

#endif