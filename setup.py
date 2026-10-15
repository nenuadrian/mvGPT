# Builds the C++ BPE tokenizer as a Python module, bpe, next to this file:
#   python setup.py build_ext --inplace
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="bpe",
    ext_modules=[Pybind11Extension("bpe", ["bpe/bpe.cpp"], cxx_std=17, extra_compile_args=["-O3"])],
    cmdclass={"build_ext": build_ext},
)
