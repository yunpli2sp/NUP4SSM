# FFBDD: Forward Filtering Backward Dual Deciding

This repository contains code to reproduce experiment in the paper:
Yun-Peng Li, H.-A. Loeliger, “Dual NUP Representations and Min-Maximization in Factor Graphs,” 
(to be appeared) in 2025 IEEE International Symposium on Information Theory (ISIT 2025),
June 2025. 

The library is implemented with C, and interface is provided for python users. You can use FFBDD
to compute MAP estimation for state space models with non-Gaussian or linearly constrained outputs.

## Dependence

For python: numpy, matplotlib, cython, jupyter-notebook.

For C: cblas, clapack.

## create dynamic library for python user

### C complie

          cc -Wall -g -O -fPIC -c dual_utils.c -o dual_utils.o 
          cc -Wall -g -O -fPIC -c ffbdd_smoothing.c -o ffbdd_smoothing.o
          cc -shared -o libnup4ssm.so *.o  -lcblas -lblas -llapack

### cython complie

          export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
          python setup.py build_ext --inplace

### output

          pyNUP4SSM.cpython-311-darwin.so


## demo 

          run demo
               1.) random linear model predictive control
           using command: 
               python ./LinearMPC.py
         
           

## Author 
   Yun-Peng Li (yunpli@isi.ee.ethz.ch)
