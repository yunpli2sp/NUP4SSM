  Copyright (C) 2024 Yun-Peng Li, Hans-Andrea Loeliger.
 
  Date: 2024-2-13
  Author: Yunpeng Li (yunpli@isi.ee.ethz.ch)
  Brief: Performing MAP estimation for linear non-Gaussian state space 
models using
         BFFD (backward filtering forward deciding) algorithm. Two demos 
in papers
         are given:
             1). regression with non-Gaussian inputs on 'global warming 
datasets'.
             2). linear quadratic constrained control for two dimensional 
object moving.
  Cite: Y.-P Li, H.-A. Loeliger, “Backward Filtering Forward Deciding in 
Linear Non-Gaussian 
         State Space Models,” (to be appeared) in 2024 International 
Conference on Artificial 
         Intelligence and Statistics (AISTATS2024), May 2024.
  Dependence:---python
                   ---- numpy
                   ---- matplotlib
                   ---- cython
             ---C
                   ---- clbas
                   ---- clapack
  
  Complie:
          ---C
          
          cc -Wall -g -O -fPIC -c utils.c -o utils.o 
          cc -Wall -g -O -fPIC -c bffd_smoothing.c -o bffd_smoothing.o
          cc -shared -o libnup4ssm.so *.o  -lcblas -lblas -llapack
          
          ---cython
          
          export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
          python setup.py build_ext --inplace
