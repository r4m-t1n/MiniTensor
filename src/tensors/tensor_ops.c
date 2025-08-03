#include <stdio.h>
#include <stdlib.h>
#include "types.h"

#define TENSOR_BINARY_OP(type, op, a, b, out, size) \
    do {                                            \
        for (int i = 0; i < size; i++) {            \
            ((type*)(out))[i] = ((type*)(a))[i] op ((type*)(b))[i]; \
        }                                           \
    } while (0)


int check_tensor_validity(struct Tensor* a, struct Tensor* b){
    if (a == NULL || b == NULL) {
        fprintf(stderr, "ERROR: Cannot do operations on NULL tensors\n");
        return -1;
    }

    if (a->dtype != b->dtype){
        fprintf(stderr, "ERROR: DTypes are not the same: %d and %d\n", a->dtype, b->dtype);
        return -1;
    }

    if (a->ndim != b->ndim){
        fprintf(stderr, "ERROR: Shapes are not the same: %d and %d\n", a->ndim, b->ndim);
        return -1;
    }

    for (int i=0; i<a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "ERROR: Shapes are not the same at dimension %d: %d and %d\n", i, a->shape[i], b->shape[i]);
            return -1;
        }
    }
    return 0;
}


struct Tensor* tensor_add(struct Tensor* a, struct Tensor* b){
    int is_tensor_valid = check_tensor_validity(a, b);

    if (is_tensor_valid == -1){
        return NULL;
    }

    struct Tensor* result_tensor = create_tensor(a->shape, a->ndim, a->dtype);

    if (a->dtype == DTYPE_INT){
        TENSOR_BINARY_OP(int, +, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_FLOAT){
        TENSOR_BINARY_OP(float, +, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_DOUBLE){
        TENSOR_BINARY_OP(double, +, a->data, b->data, result_tensor->data, a->size);
    } else {
        fprintf(stderr, "ERROR: Invalid DType: %d\n", a->dtype);
        return NULL;
    }

    return result_tensor;
}


struct Tensor* tensor_sub(struct Tensor* a, struct Tensor* b){
    int is_tensor_valid = check_tensor_validity(a, b);

    if (is_tensor_valid == -1){
        return NULL;
    }

    struct Tensor* result_tensor = create_tensor(a->shape, a->ndim, a->dtype);

    if (a->dtype == DTYPE_INT){
        TENSOR_BINARY_OP(int, -, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_FLOAT){
        TENSOR_BINARY_OP(float, -, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_DOUBLE){
        TENSOR_BINARY_OP(double, -, a->data, b->data, result_tensor->data, a->size);
    } else {
        fprintf(stderr, "ERROR: Invalid DType: %d\n", a->dtype);
        return NULL;
    }
    return result_tensor;
}

struct Tensor* tensor_mul(struct Tensor* a, struct Tensor* b){
    int is_tensor_valid = check_tensor_validity(a, b);

    if (is_tensor_valid == -1){
        return NULL;
    }

    struct Tensor* result_tensor = create_tensor(a->shape, a->ndim, a->dtype);

    if (a->dtype == DTYPE_INT){
        TENSOR_BINARY_OP(int, *, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_FLOAT){
        TENSOR_BINARY_OP(float, *, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_DOUBLE){
        TENSOR_BINARY_OP(double, *, a->data, b->data, result_tensor->data, a->size);
    } else {
        fprintf(stderr, "ERROR: Invalid DType: %d\n", a->dtype);
        return NULL;
    }

    return result_tensor;
}


struct Tensor* tensor_div(struct Tensor* a, struct Tensor* b){
    int is_tensor_valid = check_tensor_validity(a, b);

    if (is_tensor_valid == -1){
        return NULL;
    }

    struct Tensor* result_tensor = create_tensor(a->shape, a->ndim, a->dtype);

    if (a->dtype == DTYPE_INT){
        for (int i = 0; i < a->size; i++) {
            int denominator = ((int*)b->data)[i];
            if (denominator == 0) {
                fprintf(stderr, "ERROR: Division by zero at index %d\n", i);
                free_tensor(result_tensor);
                return NULL;
            }
            ((int*)result_tensor->data)[i] = ((int*)a->data)[i] / denominator;
        }
    } else if (a->dtype == DTYPE_FLOAT){
        TENSOR_BINARY_OP(float, /, a->data, b->data, result_tensor->data, a->size);
    } else if (a->dtype == DTYPE_DOUBLE){
        TENSOR_BINARY_OP(double, /, a->data, b->data, result_tensor->data, a->size);
    } else {
        fprintf(stderr, "ERROR: Invalid DType: %d\n", a->dtype);
        return NULL;
    }
    return result_tensor;
}