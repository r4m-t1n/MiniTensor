#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "losses/mse.h"

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cpp, m) {
    m.doc() = "MiniTensor! WwWwWoWwWwW";

    auto m_float = m.def_submodule("float32");
    py::class_<Tensor<float>>(m_float, "Tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&, bool>(),
             py::arg("data_vec"), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<int>&, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<float*, const std::vector<int>&, bool>(),
             py::arg("data_ptr"), py::arg("shape"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Tensor<float>::data)
        .def_readwrite("shape", &Tensor<float>::shape)
        .def_readwrite("ndim", &Tensor<float>::ndim)
        .def_readwrite("size", &Tensor<float>::size)
        .def_readwrite("stride", &Tensor<float>::stride)
        .def_readwrite("requires_grad", &Tensor<float>::requires_grad)
        .def_readwrite("grad", &Tensor<float>::grad)
        .def("backward", &Tensor<float>::backward)
        .def("to_vector", &to_vector<float>)
        .def("to_nested", &to_nested_wrapper<float>)
        .def("__repr__", &tensor_repr<float>)
        .def("__matmul__", &mat_mul<float>)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self);

    m_float.def("mat_mul", &mat_mul<float>);
    m_float.def("add", &tensor_add<float>);
    m_float.def("sub", &tensor_sub<float>);
    m_float.def("mul", &tensor_mul<float>);
    m_float.def("div", &tensor_div<float>);

    m_float.def("add_scalar", &tensor_scalar_add<float, float>);
    m_float.def("sub_scalar", &tensor_scalar_sub<float, float>);
    m_float.def("mul_scalar", &tensor_scalar_mul<float, float>);
    m_float.def("div_scalar", &tensor_scalar_div<float, float>);

    m_float.def("mse_loss", &mse_loss<float>);

    auto m_double = m.def_submodule("float64");
    py::class_<Tensor<double>>(m_double, "Tensor")
        .def(py::init<const std::vector<double>&, const std::vector<int>&, bool>(),
             py::arg("data_vec"), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<int>&, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<double*, const std::vector<int>&, bool>(),
             py::arg("data_ptr"), py::arg("shape"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Tensor<double>::data)
        .def_readwrite("shape", &Tensor<double>::shape)
        .def_readwrite("ndim", &Tensor<double>::ndim)
        .def_readwrite("size", &Tensor<double>::size)
        .def_readwrite("stride", &Tensor<double>::stride)
        .def_readwrite("requires_grad", &Tensor<double>::requires_grad)
        .def_readwrite("grad", &Tensor<double>::grad)
        .def("backward", &Tensor<double>::backward)
        .def("to_vector", &to_vector<double>)
        .def("to_nested", &to_nested_wrapper<double>)
        .def("__repr__", &tensor_repr<double>)
        .def("__matmul__", &mat_mul<double>)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(py::self / double())
        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self);

    m_double.def("mat_mul", &mat_mul<double>);
    m_double.def("add", &tensor_add<double>);
    m_double.def("sub", &tensor_sub<double>);
    m_double.def("mul", &tensor_mul<double>);
    m_double.def("div", &tensor_div<double>);

    m_double.def("add_scalar", &tensor_scalar_add<double, double>);
    m_double.def("sub_scalar", &tensor_scalar_sub<double, double>);
    m_double.def("mul_scalar", &tensor_scalar_mul<double, double>);
    m_double.def("div_scalar", &tensor_scalar_div<double, double>);

    m_double.def("mse_loss", &mse_loss<double>);

    auto m_int = m.def_submodule("int32");
    py::class_<Tensor<int>>(m_int, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, bool>(),
             py::arg("data_vec"), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<int>&, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<int*, const std::vector<int>&, bool>(),
             py::arg("data_ptr"), py::arg("shape"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Tensor<int>::data)
        .def_readwrite("shape", &Tensor<int>::shape)
        .def_readwrite("ndim", &Tensor<int>::ndim)
        .def_readwrite("size", &Tensor<int>::size)
        .def_readwrite("stride", &Tensor<int>::stride)
        .def_readwrite("requires_grad", &Tensor<int>::requires_grad)
        .def_readwrite("grad", &Tensor<int>::grad)
        .def("backward", &Tensor<int>::backward)
        .def("to_vector", &to_vector<int>)
        .def("to_nested", &to_nested_wrapper<int>)
        .def("__repr__", &tensor_repr<int>)
        .def("__matmul__", &mat_mul<int>)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + int())
        .def(py::self - int())
        .def(py::self * int())
        .def(py::self / int())
        .def(int() + py::self)
        .def(int() - py::self)
        .def(int() * py::self)
        .def(int() / py::self);

    m_int.def("mat_mul", &mat_mul<int>);
    m_int.def("add", &tensor_add<int>);
    m_int.def("sub", &tensor_sub<int>);
    m_int.def("mul", &tensor_mul<int>);
    m_int.def("div", &tensor_div<int>);

    m_int.def("add_scalar", &tensor_scalar_add<int, int>);
    m_int.def("sub_scalar", &tensor_scalar_sub<int, int>);
    m_int.def("mul_scalar", &tensor_scalar_mul<int, int>);
    m_int.def("div_scalar", &tensor_scalar_div<int, int>);

    m_int.def("mse_loss", &mse_loss<int>);
}