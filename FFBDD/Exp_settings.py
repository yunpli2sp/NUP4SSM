#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:58:16 2024

@author: yunpengli
"""

import numpy as np

def H_matrix_function(J,D,rho):
    cov = rho*np.ones((D,D),dtype=float)
    np.fill_diagonal(cov, 1.)
    return cov/J


def generate_random_matrix(J,D,cov):
    return np.random.multivariate_normal(np.zeros((D,)),cov,(J,))

def generate_noisy_observations(A,B,C,N,M,P,K,msgf_m_X1, msgf_V_X1, 
                                msgf_m_Uns, msgf_V_Uns, ratio):
    
    y_obs = np.zeros((N,K,1),dtype=float)
    lower_bounds = np.zeros((N,K),dtype=float)
    upper_bounds = np.zeros((N,K),dtype=float)
    m_X1 = np.random.multivariate_normal(msgf_m_X1[:,0],msgf_V_X1)[:,np.newaxis]
    m_Xn = m_X1
    for n in range(N): 
        m_Yn = C@m_Xn
        y_obs[n,:,0] = m_Yn[:,0] + np.multiply(np.random.uniform(-ratio,ratio,K),np.abs(m_Yn[:,0]))
        lower_bounds[n,:] = y_obs[n,:,0]-ratio*np.abs(y_obs[n,:,0])
        upper_bounds[n,:] = y_obs[n,:,0]+ratio*np.abs(y_obs[n,:,0]) 
        m_Un = np.random.multivariate_normal(msgf_m_Uns[:,0],msgf_V_Uns)[:,np.newaxis]
        m_Xn = A@m_Xn + B@m_Un
    return y_obs, lower_bounds, upper_bounds


def generate_parameters(N,M,P,K,rho):
    cov_A = H_matrix_function(N,M,rho)
    A = generate_random_matrix(M,M,cov_A)
    cov_B = H_matrix_function(N,P,rho)
    B = generate_random_matrix(M,P,cov_B)
    cov_C = H_matrix_function(N,M,rho)
    C = generate_random_matrix(K,M,cov_C)
    msgf_m_X1 = np.zeros((M,1),dtype=float)
    msgf_V_X1 = cov_A
    msgf_m_Uns = np.zeros((P,1),dtype=float)
    msgf_V_Uns = H_matrix_function(P,P,rho)
    msgb_m_Zns = np.zeros((K,1),dtype=float)
    msgb_V_Zns = H_matrix_function(K,K,rho)
    return A, B, C, msgf_m_X1, msgf_V_X1, msgf_m_Uns, msgf_V_Uns, msgb_m_Zns, msgb_V_Zns