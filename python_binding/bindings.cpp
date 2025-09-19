#include <memory>
#include <variant>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensors/tensors.h"
#include "losses/losses.h"
#include "nn/activations/activations.h"
#include "nn/layers/layers.h"
#include "nn/initializers/initializers.h"

namespace py = pybind11;

template<typename T>
void define_bindings_for_type(py::module_& m, const std::string& type_name) {
     auto m_type = m.def_submodule(type_name.c_str());

     py::class_<Tensor<T>, std::shared_ptr<Tensor<T>>>(m_type, "Tensor")
          .def(py::init([](const std::vector<T>& data_vec, const std::vector<int>& shape, bool req_grad) {
               return std::make_shared<Tensor<T>>(data_vec, shape, req_grad);
          }), py::arg("data_vec"), py::arg("shape"), py::arg("requires_grad") = false)
          
          .def(py::init([](const std::vector<int>& shape, bool req_grad) {
               return std::make_shared<Tensor<T>>(shape, req_grad);
          }), py::arg("shape"), py::arg("requires_grad") = false)

          .def_readwrite("shape", &Tensor<T>::shape)
          .def_readwrite("requires_grad", &Tensor<T>::requires_grad)
          .def_readwrite("grad", &Tensor<T>::grad)

          .def("backward", &Tensor<T>::backward)
          .def("zero_grad", &Tensor<T>::zero_grad) 
          .def("to_vector", &to_vector<T>)
          .def("to_nested", &to_nested_wrapper<T>)
          .def("set_data", &Tensor<T>::set_data, py::arg("other"))

          .def("__repr__", &tensor_repr<T>)

          .def("__matmul__", [](std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) { return mat_mul(a, b); })
          .def("__add__", [](std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) { return tensor_add(a, b); })
          .def("__sub__", [](std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) { return tensor_sub(a, b); })
          .def("__mul__", [](std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) { return tensor_mul(a, b); })
          .def("__truediv__", [](std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b) { return tensor_div(a, b); })

          .def("__add__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_add(a, scalar); })
          .def("__sub__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_sub(a, scalar); })
          .def("__mul__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_mul(a, scalar); })
          .def("__truediv__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_div(a, scalar); })
          
          .def("__radd__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_add(a, scalar); })
          .def("__rsub__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return scalar_tensor_sub(scalar, a); })
          .def("__rmul__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return tensor_scalar_mul(a, scalar); })
          .def("__rtruediv__", [](std::shared_ptr<Tensor<T>> a, T scalar) { return scalar_tensor_div(scalar, a); });

     m_type.def("mse_loss", &mse_loss<T>);
     m_type.def("mae_loss", &mae_loss<T>);
     m_type.def("bce_loss", &bce_loss<T>);
     m_type.def("relu", &relu<T>);
     m_type.def("sum", &sum<T>, py::arg("tensor"), py::arg("axis") = -1);
     m_type.def("mean", &mean<T>, py::arg("tensor"), py::arg("axis") = -1);
     m_type.def("max", &max<T>, py::arg("tensor"), py::arg("axis") = -1);
     m_type.def("min", &min<T>, py::arg("tensor"), py::arg("axis") = -1);

     py::class_<Constant_Val<T>, std::shared_ptr<Constant_Val<T>>>(m_type, "Constant").def(py::init<T>());

     if constexpr (std::is_floating_point_v<T>) {
          m_type.def("tanh", &tanh_fn<T>);
          py::class_<HeNormal<T>, std::shared_ptr<HeNormal<T>>>(m_type, "HeNormal").def(py::init<>());
          py::class_<XavierUniform<T>, std::shared_ptr<XavierUniform<T>>>(m_type, "XavierUniform").def(py::init<>());
     }

     using Initializer = typename std::conditional_t<
        std::is_floating_point_v<T>,
        std::variant<std::shared_ptr<HeNormal<T>>, std::shared_ptr<XavierUniform<T>>, std::shared_ptr<Constant_Val<T>>>,
        std::variant<std::shared_ptr<Constant_Val<T>>>
    >;
    py::class_<Initializer>(m_type, "Initializer");

     auto linear_cls = py::class_<Linear<T>, std::shared_ptr<Linear<T>>>(m_type, "Linear");
     
     if constexpr (std::is_floating_point_v<T>) {
          linear_cls.def(py::init([](int in, int out, Initializer w_init, Initializer b_init) {
               return std::make_shared<Linear<T>>(in, out, w_init, b_init);
          }), py::arg("input_features"), py::arg("output_features"),
             py::arg("weight_init") = std::make_shared<HeNormal<T>>(),
             py::arg("bias_init") = std::make_shared<Constant_Val<T>>(0.0f));

          m_type.def("sqrt", &tensor_sqrt<T, T>);
          m_type.def("log", &tensor_log<T, T>);
          m_type.def("exp", &tensor_exp<T, T>);
          m_type.def("pow", &tensor_pow<T, T>);
          m_type.def("sin", &tensor_sin<T, T>);
          m_type.def("cos", &tensor_cos<T, T>);
          m_type.def("tan", &tensor_tan<T, T>);
     } else {
          linear_cls.def(py::init([](int in, int out, Initializer w_init, Initializer b_init) {
               return std::make_shared<Linear<T>>(in, out, w_init, b_init);
          }), py::arg("input_features"), py::arg("output_features"),
             py::arg("weight_init") = std::make_shared<Constant_Val<T>>(1),
             py::arg("bias_init") = std::make_shared<Constant_Val<T>>(0));

          m_type.def("sqrt", &tensor_sqrt<T, float>);
          m_type.def("log", &tensor_log<T, float>);
          m_type.def("exp", &tensor_exp<T, float>);
          m_type.def("pow", &tensor_pow<T, float>);
          m_type.def("sin", &tensor_sin<T, float>);
          m_type.def("cos", &tensor_cos<T, float>);
          m_type.def("tan", &tensor_tan<T, float>);
     }

     linear_cls.def("forward", &Linear<T>::forward);
     linear_cls.def("parameters", &Linear<T>::parameters);
     linear_cls.def("__repr__", &linear_repr<T>);
     linear_cls.def("__call__", &Linear<T>::forward);
}

PYBIND11_MODULE(minitensor_cpp, m) {
     m.doc() = "MiniTensor! WwWwWoWwWwW";

     define_bindings_for_type<float>(m, "float32");
     define_bindings_for_type<double>(m, "float64");
     define_bindings_for_type<int>(m, "int32");
}