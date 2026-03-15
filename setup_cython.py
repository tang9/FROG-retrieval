"""Build script for Cython extension modules."""

import sys

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "retrieval_class.retrievers.rana_cython",
        ["retrieval_class/retrievers/rana_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]

# Allow running `python setup_cython.py` without extra args.
if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

setup(
    packages=["retrieval_class"],
    package_dir={"": "."},
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
