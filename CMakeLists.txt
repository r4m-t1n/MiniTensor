cmake_minimum_required(VERSION 3.14)
project(minitensor)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(minitensor python_binding/bindings.cpp)

target_include_directories(minitensor PRIVATE ${PROJECT_SOURCE_DIR}/minitensor)