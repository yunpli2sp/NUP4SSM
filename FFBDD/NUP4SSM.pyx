#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:39:43 2024

@author: Yunpeng Li
"""


from libc.stdio cimport printf

from cython cimport floating


cimport numpy as cnp
import numpy as np

ctypedef cnp.float64_t DOUBLE
ctypedef cnp.uint32_t UINT32_t
from cython cimport floating
cnp.import_array()




cdef extern from "ffbdd_smoothing.h":

    int iterated_ffbdd_nonGaussian_output(int choice, int is_constrained, int N, int M, int L, int K, double *lower_bounds, double * upper_bounds, 
                                          double beta, double *A, double *B, double *C, double *msgf_m_X1, double *msgf_V_X1, double * msgf_W_X1, 
                                          double *dual_xi_Xf, double *dual_W_Xf, double * msgf_m_Uns, double * msgf_V_Uns, double * msgf_W_Uns, 
                                          int maxit, double obj_tol, double init_gamma, double *m_X1, double *m_Us, double *m_Ys, double *objs);





def pyFFBDD_output_estimation(int choice, int is_constrained, double[:,::1] lower_bounds, double[:,::1] upper_bounds, double beta,double[:,::1] A, 
                             double[:,::1] B, double[:,::1] C, double[::1] msgf_m_X1, double [:,::1] msgf_V_X1, double [:,::1]msgf_W_X1, 
                             double[::1] dual_xi_Xf, double[:,::1] dual_W_Xf, double[::1] msgf_m_Uns,double [:,::1] msgf_V_Uns, double [:,::1] msgf_W_Uns, 
                             int maxit, double obj_tol, double init_gamma):
    
    cdef int N = lower_bounds.shape[0]
    cdef int M = A.shape[0]
    cdef int L = B.shape[1]
    cdef int K = C.shape[0]
    cdef int total_iter = -1

    


    
    
    cdef double[::1] m_X1 = np.zeros((M,),dtype=np.float64)
    cdef double[:,::1] m_Us = np.zeros((N,L),dtype=np.float64)
    cdef double[:,::1] m_Ys = np.zeros((N,K),dtype=np.float64)
    cdef double[::1] objs = np.zeros((maxit+1,),dtype=np.float64)
    
    #printf("cython start")
    total_iter = iterated_ffbdd_nonGaussian_output(choice,is_constrained, N,M,L,K, &lower_bounds[0,0], &upper_bounds[0,0], beta, &A[0,0],&B[0,0],&C[0,0],
                                     &msgf_m_X1[0], &msgf_V_X1[0,0],&msgf_W_X1[0,0], &dual_xi_Xf[0],&dual_W_Xf[0,0], &msgf_m_Uns[0], &msgf_V_Uns[0,0], 
                                     &msgf_W_Uns[0,0], maxit, obj_tol, init_gamma, &m_X1[0],&m_Us[0,0], &m_Ys[0,0], &objs[0])
    
    #printf("cython end")



    if(total_iter==-1):
        printf("Error occurrence")
    
    return np.asarray(m_X1), np.asarray(m_Us), np.asarray(m_Ys),np.asarray(objs), total_iter






