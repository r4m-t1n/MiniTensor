#include <memory>
#include <variant>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include "tensors/tensor.h"
#include "tensors/tensor_ops.h"
#include "losses/mse.h"
#include "nn/activations/relu.h"
#include "nn/activations/tanh.h"
#include "nn/initializers/initializers.h"
#include "nn/layers/linear.h"

namespace py = pybind11;

template<typename T>
void define_bindings_for_type(py::module_& m, const std::string& type_name) {
     auto m_type = m.def_submodule(type_name.c_str());

     py::class_<Tensor<T>>(m_type, "Tensor")
          .def(py::init<const std::vector<T>&, const std::vector<int>&, bool>(),
               py::arg("data_vec"), py::arg("shape"), py::arg("requires_grad") = false)
          .def(py::init<const std::vector<int>&, bool>(),
               py::arg("shape"), py::arg("requires_grad") = false)

          .def_readwrite("shape", &Tensor<T>::shape)
          .def_readwrite("requires_grad", &Tensor<T>::requires_grad)
          .def_readwrite("grad", &Tensor<T>::grad)

          .def("backward", &Tensor<T>::backward)
          .def("to_vector", &to_vector<T>)
          .def("to_nested", &to_nested_wrapper<T>)

          .def("__repr__", &tensor_repr<T>)
          .def("__matmul__", &mat_mul<T>)

          .def(py::self + py::self)
          .def(py::self - py::self)
          .def(py::self * py::self)
          .def(py::self / py::self)
          .def(py::self + T())
          .def(py::self - T())
          .def(py::self * T())
          .def(py::self / T())
          .def(T() + py::self)
          .def(T() - py::self)
          .def(T() * py::self)
          .def(T() / py::self);

     m_type.def("mse_loss", &mse_loss<T>);
     m_type.def("relu", &relu<T>);

     py::class_<Constant_Val<T>, std::shared_ptr<Constant_Val<T>>>(m_type, "Constant").def(py::init<T>());

     if constexpr (std::is_floating_point_v<T>) {
          m_type.def("tanh", &tanh<T>);
          py::class_<HeNormal<T>, std::shared_ptr<HeNormal<T>>>(m_type, "HeNormal").def(py::init<>());
          py::class_<XavierUniform<T>, std::shared_ptr<XavierUniform<T>>>(m_type, "XavierUniform").def(py::init<>());
     }

     py::class_<Initializer<T>>(m_type, "Initializer");

     auto linear_cls = py::class_<Linear<T>, std::shared_ptr<Linear<T>>>(m_type, "Linear");
     
     if constexpr (std::is_floating_point_v<T>) {
          linear_cls.def(py::init<int, int, Initializer<T>, Initializer<T>>(),
               py::arg("input_features"),
               py::arg("output_features"),
               py::arg("weight_init") = HeNormal<T>{},
               py::arg("bias_init") = Constant_Val<T>{0.0f});
     } else {
          linear_cls.def(py::init<int, int, Initializer<T>, Initializer<T>>(),
               py::arg("input_features"),
               py::arg("output_features"),
               py::arg("weight_init") = Constant_Val<T>{0},
               py::arg("bias_init") = Constant_Val<T>{0});
     }

     linear_cls.def("forward", &Linear<T>::forward);
     linear_cls.def("parameters", &Linear<T>::parameters, py::return_value_policy::reference_internal);
     linear_cls.def("__repr__", &linear_repr<T>);
}

PYBIND11_MODULE(minitensor_cpp, m) {
     m.doc() = "MiniTensor! WwWwWoWwWwW";

     define_bindings_for_type<float>(m, "float32");
     define_bindings_for_type<double>(m, "float64");
     define_bindings_for_type<int>(m, "int32");
}