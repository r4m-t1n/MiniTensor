#include <stdio.h>
#include <stdlib.h>

#define DEFINE_MSE(type, name)              \
type name(type y, type y_hat) {             \
    return (y - y_hat) * (y - y_hat);       \
}

DEFINE_MSE(int, mse_loss_int)
DEFINE_MSE(float, mse_loss_float)
DEFINE_MSE(double, mse_loss_double)