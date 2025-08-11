from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "minitensor.minitensor_cpp",
        ["python_binding/bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "core"
        ],
        cxx_std=17,
    ),
]

setup(
    name="minitensor",
    version="0.1",
    author="r4m-t1n",
    packages=["minitensor"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
