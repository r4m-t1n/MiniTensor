#ifndef TENSOR_OPS
#define TENSOR_OPS

int check_tensor_validity(struct Tensor* a, struct Tensor* b);

struct Tensor* tensor_add(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_sub(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_mul(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_div(struct Tensor* a, struct Tensor* b);

#endif