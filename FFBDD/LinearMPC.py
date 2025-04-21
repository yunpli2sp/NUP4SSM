#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:27:05 2024

@author: Yunpeng Li
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import cython
import cvxpy as cp



from pyNUP4SSM import pyFFBDD_output_estimation
from Exp_settings import *





def qp_solver(A,B,C,msgf_m_X1,msgf_W_X1, msgf_m_Uns, msgf_W_Uns,lower_bounds,upper_bounds,solver, maxit):
    
    N = lower_bounds.shape[0]
    M = A.shape[0]
    L = B.shape[1]
    K = C.shape[0]
    xs = cp.Variable((M, N + 1))
    us = cp.Variable((L, N))
    ys = cp.Variable((K, N))
    
    
    cost = 0.5*cp.quad_form(xs[:,0], msgf_W_X1) - ((msgf_W_X1@msgf_m_X1).T@xs[:,0])
    constr = []
    for n in range(N):
        cost += 0.5*cp.quad_form(us[:,n], msgf_W_Uns) - ((msgf_W_Uns@msgf_m_Uns).T@us[:,n])
        constr += [xs[:, n + 1] == A @ xs[:, n] + B @ us[:, n], ys[:,n] == C@xs[:, n], ys[:,n]>=lower_bounds[n,:], ys[:,n]<=upper_bounds[n,:]]




    
    problem = cp.Problem(cp.Minimize(cost), constr)
    if(solver=="OSQP" or solver =="CLARABEL"):
        problem.solve(solver= solver, verbose=False, solver_verbose=False,max_iter= maxit)
    elif(solver=="PIQP"):
        problem.solve(solver= solver, verbose=False, solver_verbose=False,max_iter= maxit,compute_timings=True)
    elif(solver=="DAQP"):
        problem.solve(solver= solver, verbose=False, solver_verbose=False,iter_limit= maxit,compute_timings=True)
    else:
        problem.solve(solver= solver, verbose=False, solver_verbose=False,max_iters= maxit)
    
    return problem.solver_stats.solve_time, problem.solver_stats.num_iters, problem.value








if __name__ == '__main__':
    
    N = 1000
    M = 40
    L = 20
    K = 20
    reps = 1
    

    rho = 0.0
    maxit = 1000
    obj_tol = 1e-8
    a = 0
    b = 0
    ratio = 0.1
    n_methods = 6

    
    np.random.RandomState(0)


    
    
    
    iter_indexs = np.arange(maxit)

    
    Exp_record_iters = np.zeros((5,reps),dtype=int)
    Exp_record_cost_funcs = np.zeros((5,reps),dtype=float)
    Exp_record_cost_funcs_per_iterations = np.zeros((5,reps,maxit+1),dtype=float)

    Exp_record_timing = np.zeros((5,reps),dtype=float)
    Exp_min_cost_funcs = np.zeros((reps),dtype=float)
    

    msgb_m_Xf = np.zeros((M,1),dtype=float) 
    msgb_W_Xf = np.zeros((M,M),dtype=float) 
    
    dual_xi_Xf = np.zeros((M,1),dtype=float) 
    dual_W_Xf = np.zeros((M,M),dtype=float) 

    
    for r in range(reps):
        A, B, C, msgf_m_X1, msgf_V_X1, msgf_m_Uns, msgf_V_Uns, msgb_m_Zns, msgb_V_Zns = generate_parameters(N,M,L,K,rho)
        msgf_W_X1 = np.linalg.inv(msgf_V_X1)
        msgf_W_Uns = np.linalg.inv(msgf_V_Uns)
        y_obs, lower_bounds, upper_bounds = generate_noisy_observations(A,B,C,N,M,L,K,msgf_m_X1, msgf_V_X1, msgf_m_Uns, msgf_V_Uns, ratio)


        
        print("==========================================random linear model predictive control============================================")



        choice = 3
        is_constrained = 1
        start_time = time.time()
        beta = 1.0
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> IFFBDD smoother <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        _, _, _, cost_funcs0, it0 = pyFFBDD_output_estimation(choice, is_constrained, lower_bounds, upper_bounds, beta, A, B, C, msgf_m_X1[:,0], msgf_V_X1, msgf_W_X1, 
                                  dual_xi_Xf[:,0], dual_W_Xf, msgf_m_Uns[:,0], msgf_V_Uns, msgf_W_Uns, maxit, obj_tol,init_gamma=1e-12)
        
        Exp_record_timing[0,r] = time.time() - start_time
        Exp_record_iters[0,r] = it0
        Exp_record_cost_funcs_per_iterations[0,r,:] = cost_funcs0.copy()
        Exp_record_cost_funcs[0,r] = cost_funcs0[it0]
        total_time = time.time() - start_time
        print("IFFBDD smoother: cost = {0}, running time ={1}, iterations ={2}, total time ={3}".format(cost_funcs0[it0],Exp_record_timing[0,r],Exp_record_iters[0,r],total_time))
     
        
        
        solver = "PIQP"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PIQP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        start_time = time.time()
        Exp_record_timing[1,r],Exp_record_iters[1,r],Exp_record_cost_funcs[1,r] = qp_solver(A,B,C,msgf_m_X1,msgf_W_X1, msgf_m_Uns, msgf_W_Uns,lower_bounds,upper_bounds,solver, maxit)
        total_time = time.time() - start_time
        print("piqp: cost = {0}, running time ={1}, total time ={2}".format(Exp_record_cost_funcs[1,r],Exp_record_timing[1,r],total_time))
        
        
        
        solver = "ECOS"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ECOS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        start_time = time.time()
        Exp_record_timing[2,r],Exp_record_iters[2,r],Exp_record_cost_funcs[2,r] = qp_solver(A,B,C,msgf_m_X1,msgf_W_X1, msgf_m_Uns, msgf_W_Uns,lower_bounds,upper_bounds,solver, maxit)
        total_time = time.time() - start_time
        print("ecos: cost = {0}, running time ={1}, total time ={2}".format(Exp_record_cost_funcs[2,r],Exp_record_timing[2,r],total_time))
    
        
        solver = "SCS"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SCS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        start_time = time.time()
        Exp_record_timing[3,r],Exp_record_iters[3,r],Exp_record_cost_funcs[3,r] = qp_solver(A,B,C,msgf_m_X1,msgf_W_X1, msgf_m_Uns, msgf_W_Uns,lower_bounds,upper_bounds,solver, maxit)
        total_time = time.time() - start_time
        print("scs: cost = {0}, running time ={1}, total time ={2}".format(Exp_record_cost_funcs[3,r],Exp_record_timing[3,r],total_time))
    
        
        
        solver = "CLARABEL"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CLARABEL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        start_time = time.time()
        Exp_record_timing[4,r],Exp_record_iters[4,r],Exp_record_cost_funcs[4,r] = qp_solver(A,B,C,msgf_m_X1,msgf_W_X1, msgf_m_Uns, msgf_W_Uns,lower_bounds,upper_bounds,solver, maxit)
        total_time = time.time() - start_time
        print("CLARABEL: cost = {0}, running time ={1}, total time ={2}".format(Exp_record_cost_funcs[4,r],Exp_record_timing[4,r],total_time))

        

        
        
        
        
        
    #plt.figure(figsize=(6,3))  
    #plt.yscale("log")
    #plt.boxplot(Exp_record_timing.T,labels=['IFFBD', 'PIQP', 'ECOS','SCS', 'CLARABEL'])
    #plt.ylabel('Running Time (s)')
    #plt.savefig("running_time.pdf", dpi=300)

    


    

        