# Builds the C++ BPE tokenizer as a Python module, bpe, next to this file:
#   python setup.py build_ext --inplace
# SANITIZE=thread (or address) builds it with that sanitizer's checks in it, to run under the sanitizer.
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

sanitize = [f"-fsanitize={os.environ['SANITIZE']}", "-g"] if os.environ.get("SANITIZE") else []

setup(
    name="bpe",
    ext_modules=[Pybind11Extension("bpe", ["bpe/bpe.cpp"], cxx_std=17, extra_compile_args=["-O3"] + sanitize,
                                   extra_link_args=sanitize)],
    cmdclass={"build_ext": build_ext},
)
