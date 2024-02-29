#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:37:27 2023

@author: yunpengli
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

    
    
    
    #sources = ["demo.pyx","tf_admm.c","tf_d.c","cs_util.c"],
    #include_dirs=[numpy.get_include(),"/usr/local/include/","/usr/local/lib/","/home/yunpli/anaconda3/envs/py/lib/R/include/"],
    #include_dirs=[numpy.get_include()],
    #library_dirs=['/usr/local/lib/'],
    #libraries=["lapack","lapacke","refblas","tmglib"]
    )

setup(
  name = 'pyNUP4SSM',    
  ext_modules = cythonize(ext_module)
  #ext_modules = ext_modules
)