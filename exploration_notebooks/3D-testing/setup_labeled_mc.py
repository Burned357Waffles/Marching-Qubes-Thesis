from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
    "labeled_marching_cubes",
    ["labeled_marching_cubes.cpp"],
    cxx_std=17,
)

setup(
    name="labeled_marching_cubes",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
