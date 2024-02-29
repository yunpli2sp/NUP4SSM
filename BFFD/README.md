# BFFD: Backward Filtering Forward Deciding

This repository contains code to reproduce several experiments in the paper:
Y.-P Li, H.-A. Loeliger, “Backward Filtering Forward Deciding in Linear Non-Gaussian State Space Models,” (to be appeared) in 2024 International Conference on Artificial Intelligence and Statistics (AISTATS 2024), May 2024.

The library is implemented with C, and interface is provided for python users.

## Dependence

For python: numpy, matplotlib, cython, jupyter-notebook.

For C: cblas, clapack.

## create dynamic library for python user

### C complie

          cc -Wall -g -O -fPIC -c utils.c -o utils.o 
          cc -Wall -g -O -fPIC -c bffd_smoothing.c -o bffd_smoothing.o
          cc -shared -o libnup4ssm.so *.o  -lcblas -lblas -llapack

### cython complie

          export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
          python setup.py build_ext --inplace

### output

          pyNUP4SSM.cpython-311-darwin.so


## demo in jupyter-notebook

          open corresponding ipynb document, and run two demos
               1.) regression with non-Gaussian inputs on 'global warming datasets',
               2.) linear quadratic constrained control for two dimensional object moving.
           

## Author 
   Yunpeng Li (yunpli@isi.ee.ethz.ch)
