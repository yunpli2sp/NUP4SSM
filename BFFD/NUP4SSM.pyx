#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:40:38 2023

@author: yunpengli
"""

from libc.stdio cimport printf

from cython cimport floating


cimport numpy as cnp
import numpy as np

ctypedef cnp.float64_t DOUBLE
ctypedef cnp.uint32_t UINT32_t
from cython cimport floating
cnp.import_array()



cdef extern from "bffd_smoothing.h": 
    int bffd_input_estimation(int choice, double *NUP_parameters, double beta, int N, int M, int L, int K, int max_valid_index,double *A, 
                              double *B, double *C, double *y_obs,double *msgf_Xi_X1, double *msgf_W_X1, double *msgb_Xi_Xf, double *msgb_W_Xf,
                              double *msgb_W_Yn,int maxit, double obj_tol,double eps, double *m_X1, double *m_Us, double *m_Ys, 
                              double *objs)
    
    int bffd_constrained_control(int choice, double * NUP_parameters, int N, int M, int L,double * A, double * B ,
                                 double *Q, double *Qf, double * x1, double *xf,int maxit, double obj_tol, double eps,
                                 double *m_Xs,double *m_Us,double *objs);
    
    

    

    

def pyBFFD_input_estimation(int choice, double[::1] NUP_parameters, double beta, int max_valid_index,double[:,::1] A, double [:,::1] B, double [:,::1] C,
                            double [:,::1] y_obs, double [:,::1] msgb_W_Yn, double[::1] msgf_Xi_X1, double[:,::1] msgf_W_X1,
                            double[::1] msgb_Xi_Xf, double[:,::1] msgb_W_Xf,int maxit):
    
    """Cython version of the BFFD smoothing for Linear Non-Gaussian SSM with NUP inputs

        Considering the state space model
        Y_{n} = CX_{n},                      
        tilde{Y}_{n} = Y_{n} + Z_{n},       
        X_{n+1} = AX_{n} + BU_{n},          
        with non-Gaussian inputs U_{n}, and zero Gaussian (measurement) noise Z_{n}.
        
        Given the observations y_obs, the covariance matrix msgb_W_Yn of tilde{Y}_{n},
        as well as the choice of NUP priors imposed on inputs U_{n}, we want to obtain
        the estimated m_Us of inputs U_{n} and the initial state X_{1}'s estimation m_X1, 
        then we use them to reconstruct the pure ouputs $m_Ys$ for Y_{n} (n=1,...,N).
        
    
    Inputs
    ------
    choice: int 
            the choice of NUP prior imposed on intputs U_{n}.
            
                    The selection of NUP priors and their parameters.
            =======================================================================
                choice         NUP prior          NUP_parameters[0],[1]
            =======================================================================    
                  0            Laplace/L1               /,/
                  1            Hubber loss          r (threshold),/
                  2            Hinge loss           a (lower bound),/
                  3            Vapnik loss       a (lower bound),b (upper bound)
                  4            Plain NUV               /,/ 
            =======================================================================
    
    NUP_parameters: ndarray of shape (2,)
                    the parameters of selected NUP prior.
            
    beta: double
          the non-negative scale parameter for NUP priors.
          
    max_valid_index: int
          when the final state is unknown,it is the maxmum index 
          of inputs which contributes to the y_obs. default value
          is N-1.
    
    A: ndarray of shape (M,M)
       the state-transition matrix.
    
    B: ndarray of shape (M,L)
       the input matrix.
    
    C: ndarray of shape (K,M)
       the output matrix.
    
    y_obs: ndarry of shape (N,K)
           the values of output observations.
    
    msgb_W_Yn: ndarray of shape (K,K)
               the precision matrix of Gaussian noise Z_{n} (or tilde{Y_{n}}).
   
    msgf_Xi_X1: ndarray of shape (M,)
                the (forward) precision mean of initial state X_{1}, default is 
                zero vector.
                
    msgf_W_X1: ndarry of shape (M,M)
               the (forward) precision matrix of initial state X_{1}, default is
               zero matrix.
               
    msgf_Xi_Xf: ndarray of shape (M,)
                the (backward) precision mean of final state X_{N+1}, default is 
                zero vector.
                    
    msgf_W_Xf: ndarry of shape (M,M)
              the (backward) precision matrix of final state X_{N+1}, default is
              zero matrix.

    
    maxit: int
           maximum number of iterations in optimization.
    
    
    
        
    Returns
    -------
    m_X1: ndarray of shape (M,)
          the estimation of initial state X_{1}.
          
    m_Us: ndarray of shape (N,L)
          the estimation of SSM inputs U_{n}.
    
    m_Ys: ndarray of shape (N,K)
          the estimation of SSM outputs Y_{n}.
          
    objs: ndarray of shape (maxit,)
          the value of cost fuction along the iteration.
          
    total_iter: int
          the total number of iterations consumed during optimization.
    """
    
    cdef int N = y_obs.shape[0]
    cdef int M = A.shape[0]
    cdef int L = B.shape[1]
    cdef int K = C.shape[0]
    cdef double obj_tol = 1e-6
    cdef double eps = 1e-15
    cdef int total_iter = -1;

    


    
    
    cdef double[::1] m_X1 = np.zeros((M,),dtype=np.float64)
    cdef double[:,::1] m_Us = np.zeros((N,L),dtype=np.float64)
    cdef double[:,::1] m_Ys = np.zeros((N,K),dtype=np.float64)
    cdef double[::1] objs = np.zeros((maxit,),dtype=np.float64)
    
    
    total_iter = bffd_input_estimation(choice,&NUP_parameters[0],beta, N, M, L,K,max_valid_index ,&A[0,0], &B[0,0], &C[0,0], &y_obs[0,0], 
                                     &msgf_Xi_X1[0], &msgf_W_X1[0,0], &msgb_Xi_Xf[0], &msgb_W_Xf[0,0], &msgb_W_Yn[0,0], maxit, obj_tol, eps, &m_X1[0], &m_Us[0,0], 
                                     &m_Ys[0,0], &objs[0])

    

    if(total_iter==-1):
        printf("Error occurrence")
    
    return np.asarray(m_X1), np.asarray(m_Us), np.asarray(m_Ys),np.asarray(objs), total_iter





def pyBFFD_constrained_control(int choice, double[::1] NUP_parameters, int N, double[:,::1] A, double[:,::1] B,
                               double[:,::1] Q, double[:,::1] Qf, double[::1] x1, double[::1] xf, int maxit):
    
    """Cython version of the BFFD smoothing for Linear quadratic constrained control

        Considering the state space model      
        X_{n+1} = AX_{n} + BU_{n},          
        driven by linear constrained inputs U_{n}.
        
        Given the initial state x1, final state xf, state cost matrix Q, final state cost matrix Qf,
        time horizon N, as well as the choice of linear constraints imposed on inputs U_{n}, we want 
        to steer the state X_{n} from initial state x1 to final state xf using inputs U_{n} (n=1,...,N).
        
    
    Inputs
    ------
    choice: int 
            the choice of linear constraints imposed on intputs U_{n}.
            
                    The selection of NUP priors and their parameters.
            =======================================================================
                choice         NUP prior          NUP_parameters[0],[1]
                               (constraint)
            =======================================================================    
                  2            Hinge loss        a (lower bound),/
                               (halfspace)
                  3            Vapnik loss       a (lower bound),b (upper bound)
                               (box)
            =======================================================================
    
    NUP_parameters: ndarray of shape (2,)
                    the parameters of selected NUP prior.
            
    N: int
       the size of time horizon.
    
    A: ndarray of shape (M,M)
       the state-transition matrix.
    
    B: ndarray of shape (M,L)
       the input matrix.
    
    Q: ndarray of shape (M,M)
       the state cost matrix.
    
    Qf: ndarray of shape (M,M)
       the final state cost matrix.
    
    x1: ndarry of shape (M,)
        the value of initial state.
        
    xf: ndarry of shape (M,)
        the value of final state.
    
    maxit: int
           maximum number of iterations in optimization.
    
    
        
    Returns
    -------
    m_Xs: ndarray of shape (N+1,M)
          the estimation of SSM states X_{n}.
          
    m_Us: ndarray of shape (N+1,L)
          the estimation of SSM inputs U_{n}.
    
          
    objs: ndarray of shape (maxit,)
          the value of cost fuction along the iteration.
          
    total_iter: int
          the total number of iterations consumed during optimization.
    """
    
    cdef int M = A.shape[0]
    cdef int L = B.shape[1]

    cdef double obj_tol = 1e-6
    cdef double eps = 1e-15
    cdef int total_iter = -1;
    
    printf("choice = %d, NUP_parameters[0] = %f, NUP_parameters[1] = %f\n",choice,NUP_parameters[0],NUP_parameters[1]);

    


    
    
    cdef double[:,::1] m_Xs = np.zeros((N+1,M),dtype=np.float64)
    cdef double[:,::1] m_Us = np.zeros((N,L),dtype=np.float64)
    cdef double[::1] objs = np.zeros((maxit,),dtype=np.float64)
    
    
    
    total_iter = bffd_constrained_control(choice, &NUP_parameters[0], N, M, L, &A[0,0], &B[0,0], &Q[0,0], &Qf[0,0], 
                                              &x1[0], &xf[0], maxit, obj_tol, eps, &m_Xs[0,0], &m_Us[0,0], &objs[0])
    
    
    
    if(total_iter==-1):  
        printf("Error occurrence")
    
    return np.asarray(m_Xs), np.asarray(m_Us), np.asarray(objs), total_iter
    
    
    
    
    
    





