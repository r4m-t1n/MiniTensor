#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "losses/mse.h"

namespace py = pybind11;

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

PYBIND11_MODULE(minitensor_cpp, m) {
    m.doc() = "MiniTensor! WwWwWoWwWwW";

    auto m_float = m.def_submodule("float32");
    py::class_<Tensor<float>>(m_float, "Tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&>())
        .def_readwrite("data", &Tensor<float>::data)
        .def_readwrite("shape", &Tensor<float>::shape)
        .def_readwrite("ndim", &Tensor<float>::ndim)
        .def_readwrite("size", &Tensor<float>::size)
        .def("to_list", &get_tensor_data<float>)
        .def("__repr__", &tensor_repr<float>)
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
        .def(py::init<const std::vector<double>&, const std::vector<int>&>())
        .def_readwrite("data", &Tensor<double>::data)
        .def_readwrite("shape", &Tensor<double>::shape)
        .def_readwrite("ndim", &Tensor<double>::ndim)
        .def_readwrite("size", &Tensor<double>::size)
        .def("to_list", &get_tensor_data<double>)
        .def("__repr__", &tensor_repr<double>)
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
        .def(py::init<const std::vector<int>&, const std::vector<int>&>())
        .def_readwrite("data", &Tensor<int>::data)
        .def_readwrite("shape", &Tensor<int>::shape)
        .def_readwrite("ndim", &Tensor<int>::ndim)
        .def_readwrite("size", &Tensor<int>::size)
        .def("to_list", &get_tensor_data<int>)
        .def("__repr__", &tensor_repr<int>)
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