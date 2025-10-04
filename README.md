# MiniTensor

MiniTensor is a lightweight deep learning library built from scratch in C++ with Python bindings using `pybind11`.  
It’s designed as a minimal framework to explore the fundamentals of tensor operations, automatic differentiation, and neural network components, and not as a production-ready framework.

## Overview

The project provides a simple backend in C++ for core tensor operations and neural network components.  
Through `pybind11`, the C++ backend is exposed to Python, allowing the library to be used like a typical deep learning toolkit.

### Core Features

- Tensor operations (creation, arithmetic, broadcasting, reductions)
- Autograd engine for basic differentiable operations
- Neural network layers (for now only Linear)
- Common activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, MAE, BCE)
- Simple optimizers (for now only SGD)
- Python API mirroring frameworks like PyTorch

### Folder Structure

```
MiniTensor/
│
├── core/                 # C++ backend: tensor ops, autograd, layers, etc.
├── minitensor/           # Python frontend + compiled C++ extension
├── python_binding/       # pybind11 bindings
├── examples/             # Example Python scripts using the library
├── CMakeLists.txt        # Build configuration for C++
├── setup.py              # Python package setup
└── pyproject.toml
```

### Building the Project

#### Option 1: Build with CMake

```bash
mkdir build && cd build
cmake ..
make
```

This will build the shared library (`minitensor_cpp`) and place it inside the `minitensor/` directory.

#### Option 2: Build with setuptools

```bash
python3 setup.py build_ext --inplace
```

**Compatibility Notes**  
- MiniTensor currently does **not interoperate with NumPy arrays**.  
  All tensors must be created from Python lists (e.g., `tensor([[1, 2, 3]])`).  
- The library is **CPU-only**, it does not use your GPU or CUDA.  
  This is by design, to keep the implementation simple and educational.

Examples of usage can be found in the `examples/` directory.

### Motivation

MiniTensor was built as a personal learning project to better understand how deep learning frameworks work under the hood — from tensor operations to automatic differentiation and training loops.  
It’s not meant for production use, but as a clean, minimal, and educational implementation.

### License

This project is released under the MIT License.