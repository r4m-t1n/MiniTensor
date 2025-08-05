from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'minitensor',
        ['python_binding/bindings.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
    ),
]

setup(
    name='minitensor',
    version='0.1',
    author='r4m-t1n',
    packages=['python'],
    ext_modules=ext_modules,
)
