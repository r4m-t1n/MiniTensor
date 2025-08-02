#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"

struct Tensor* create_tensor(int* shape, int ndim, enum DType dtype){
    struct Tensor* tensor = malloc(sizeof(struct Tensor));
    if (tensor == NULL){
        fprintf(stderr, "ERROR: Failed to allocate memory for tensor\n");
        return NULL;
    }

    if (shape == NULL || ndim < 1){
        fprintf(stderr, "ERROR: Invalid shape\n");
        free(tensor);
        return NULL;
    }

    int size = 1;
    for (int i=0; i<ndim; i++){
        if (shape[i] <= 0) {
            fprintf(stderr, "ERROR: Dimension size must be positive (got %d at index %d)\n", shape[i], i);
            free(tensor);
            return NULL;
        }
        size *= shape[i];
    }

    if (dtype == DTYPE_INT){
        tensor->data = malloc(sizeof(int) * size);
    } else if (dtype == DTYPE_FLOAT){
        tensor->data = malloc(sizeof(float) * size);
    } else if (dtype == DTYPE_DOUBLE){
        tensor->data = malloc(sizeof(double) * size);
    } else {
        fprintf(stderr, "ERROR: Unsupported dtype\n");
        free(tensor);
        return NULL;
    }
    if (tensor->data == NULL){
        fprintf(stderr, "ERROR: Failed to allocate memory for data\n");
        free(tensor);
        return NULL;
    }

    tensor->shape = malloc(sizeof(int) * ndim);
    if (tensor->shape == NULL){
        fprintf(stderr, "ERROR: Failed to allocate memory for shape\n");
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    tensor->ndim = ndim;
    tensor->size = size;
    tensor->dtype = dtype;

    return tensor;
}

void free_tensor(struct Tensor* tensor){
    if (tensor->data != NULL){
        free(tensor->data);
        tensor->data = NULL;
    }

    if (tensor->shape != NULL){
        free(tensor->shape);
        tensor->shape = NULL;
    }

    free(tensor);
}