#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:38:58 2024

@author: Yunpeng Li
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_module = Extension(
    name ="pyNUP4SSM", 
    sources=["NUP4SSM.pyx"],
    include_dirs=[numpy.get_include()],
    library_dirs=["."],
    libraries=["nup4ssm"]
    )

setup(
  name = 'pyNUP4SSM',    
  ext_modules = cythonize(ext_module)
)