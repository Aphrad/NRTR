from matplotlib.pyplot import annotate
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("./swc2mat.pyx", annotate=True)
)