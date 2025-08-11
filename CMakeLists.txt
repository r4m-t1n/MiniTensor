cmake_minimum_required(VERSION 3.14)
project(minitensor_cpp)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(minitensor_cpp python_binding/bindings.cpp)

set_target_properties(minitensor_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/minitensor
)

target_include_directories(minitensor_cpp PRIVATE ${PROJECT_SOURCE_DIR}/core)