"""
Setup file for Cythonizing the triangle_hash.pyx file.
Compile with:
    python setup.py build_ext --inplace
"""

import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=[
        Extension(
            "triangle_hash", ["triangle_hash.cpp"], include_dirs=[numpy.get_include()]
        ),
    ],
)

setup(ext_modules=cythonize("triangle_hash.pyx"))
