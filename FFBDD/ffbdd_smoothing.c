
/* ffbdd_smoothing.c
 *
 * Copyright (C) 2025 Yun-Peng Li.
 *
 * Date: 2025-4-20
 * Author: Yunpeng Li (yunpli@isi.ee.ethz.ch)
 * Brief: Performing MAP estimation for linear non-Gaussian state space models using
 *        FFBDD (forward filtering backward dual deciding) algorithm. The density of 
 *        non-Gaussian outputs are assumed to be NUP (normal with unknown parameters) 
 *        representations.
 *        function 'ffbdd_input_estimation' is designed for general MAP estimation in 
 *        non-Gaussian SSM. 
 * Cite: Yun-Peng Li, H.-A. Loeliger, “Dual NUP Representations and Min-Maximization in Factor Graphs,” 
 *       (to be appeared) in 2005 IEEE International Symposium on Information Theory (ISIT 2025),
 *        June 2025.  
 */



#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h> // MacOs
//#include <cblas.h> // Linux

#include "dual_utils.h"
#include "ffbdd_smoothing.h"



double evaluate_quadratic_objective(int M, double * m_X, double * mean_X, double * W_X)
{   
    /*
    returns the value of quadratic objective: 0.5m_X*W_X*m_X.T - (W_X*mean_X).T*m_X, where 'T' is the matrix transpose operation.
    */
    
    double val = 0.0;
    double * vec_1;
    double * vec_2;
    double * vec_3;

    vec_1= (double *) malloc(M* sizeof(double));
    memset(vec_1,0,M*sizeof(double));

    vec_2= (double *) malloc(M* sizeof(double));
    memset(vec_2,0,M*sizeof(double));

    vec_3= (double *) malloc(M* sizeof(double));
    memset(vec_3,0,M*sizeof(double));

    cblas_dcopy(M,m_X,1,vec_1,1);
    cblas_dcopy(M,mean_X,1,vec_2,1);
    cblas_dscal(M,-1.,vec_2,1);
    cblas_daxpy(M,0.5,vec_1,1,vec_2,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,W_X,M,vec_2,1,0,vec_3,1);
    val = cblas_ddot(M,vec_1,1,vec_3,1);



    free(vec_1);
    free(vec_2);
    free(vec_3);

    return val;
}




/**
 * C version of the FFBDD smoothing for Linear Non-Gaussian SSM with NUP outputs
 * Considering the state space model
 * Y_{n} = CX_{n},                            
 * X_{n+1} = AX_{n} + BU_{n},          
 * with non-Gaussian outputs Y_{n} and Gaussian inputs U_{n}.
 * Given the mean vector $msgf_m_Uns$ and the covariance matrix $msgf_V_Uns$ of inputs
 * U_{n}, as well as the choice of NUP priors imposed on outputs Y_{n}, we want to obtain
 * the estimated $m_Ys$ of outputs Y_{n}, estimated $m_X1$ of initial state X_{1} and 
 * the estimated $m_Us$ of inputs U_{n}.
 * 
 * 
 * Arguments
 * ------
 * choice: int 
 *         the choice of NUP prior imposed on outputs Y_{n}.
 * 
 *                  The selection of NUP priors and their parameters.
 *          =======================================================================
 *               choice         NUP prior          lower_bounds[], upper_bounds[]
 *          =======================================================================    
 *                0            Laplace/L1               a,/
 *                1            Hinge loss I             a,/
 *                2            Hinge loss II            /,b
 *                3            Vapnik loss              a,b
 *                4            Quadratic loss           a,/ 
 *          =======================================================================
 * 
 * is_constrained: int
 *                 if is_constrained>0, we use loss function to enforce linear constraints imposed on Y_n.
 * 
 * N: double
 *    the value of sampe size.
 * 
 * M: double
 *    the dimension of states X_{n}.
 * 
 * L: double
 *    the dimension of inputs U_{n}.
 * 
 * K: double
 *    the dimension of outputs Y_{n}.
 * 
 * lower_bounds: double array of size N*K
 *               the lower_bounds or noisy observations of outputs Y_n.
 * 
 * upper_bounds: double array of size N*K
 *               the upper_bounds of outputs Y_n.
 * 
 * beta: double 
 *       the non-negative scale parameter for NUP priors.
 * 
 * A: double array of size M*M
 *    the state-transition matrix.
 * 
 * B: double array of size M*L
 *    the input matrix.
 * 
 * C: double array of size K*M
 *    the output matrix.
 * 
 * msgf_m_X1: double array of size M
 *             the (forward) mean vector of initial state X_{1}.
 * 
 * msgf_V_X1: double array of size M*M
 *            the (forward) covariance  matrix of initial state X_{1}.
 *  
 * msgf_W_X1: double array of size M*M
 *            the (forward) precision matrix of initial state X_{1}.
 * 
 * dual_xi_Xf: double array of size M
 *             the dual marginal mean vector of final state X_{N+1}, 
 *             default is zero vector.
 * 
 * dual_W_Xf: double array of size M*M
 *            the dual marginal covriance matrix of final state X_{N+1}, 
 *            default is zero matrix.
 * 
 * msgf_m_Uns: double array of size L
 *             the (forward) mean vector of inputs U_{n}.
 * 
 * msgf_V_Uns: double array of size L*L
 *             the (forward) covriance matrix of inputs U_{n}.
 * 
 * msgf_W_Uns: double array of size L*L
 *             the (forward) precision matrix of inputs U_{n}.
 * 
 * maxit: int
 *        maximum number of iterations in optimization.
 * 
 * obj_tol: double 
 *          stopping criteria tolerance
 * 
 * init_gamma: double
 *        initial vlue of gamma used in dual NUP representations.
 * 
 * m_X1: double array of size M
 *       the estimation of initial state X_{1}.
 *
 * m_Us: double array of size N*L
 *       the estimation of SSM inputs U_{n}.
 * 
 * m_Ys: double array of size N*L
 *       the estimation of SSM outputs Y_{n}.
 * 
 * objs: double array of size maxit
 *       the value of cost fuction along the iteration.
 * 
 * Return
 * -------
 * total_iter: int
 *             the total number of iterations consumed during optimization.
 */

int iterated_ffbdd_nonGaussian_output(int choice, int is_constrained, int N, int M, int L, int K, double *lower_bounds, double * upper_bounds, double beta, 
double *A, double *B, double *C, double *msgf_m_X1, double *msgf_V_X1, double * msgf_W_X1,double *dual_xi_Xf, double *dual_W_Xf, double * msgf_m_Uns,
double * msgf_V_Uns, double * msgf_W_Uns, int maxit, double obj_tol, double init_gamma, double *m_X1, double *m_Us, double *m_Ys, double *objs){

    printf("IFFBDD: choice = %d, is_constrained = %d\n",choice,is_constrained);

    
    int i = 0;
    int n = 0;
    int k = 0;
    int it = 0;
    double Gnk = 0;
    double gnk = 0;
    double tmp_val = 0;
    double dual_marginal_Ynk = 0;
    double obj_val = 0;
    double variation = 0;

    double msgf_m_Ynk = 0.;
    double msgf_xi_Ynk = 0;
    double msgf_W_Ynk = 0;

    double gamma_next = 0;





    double m_Ynk = 0.;
    double constraint_variation = 0.;
   


    double * m_Xn = NULL;
    double * msgf_V_Ynks = NULL;
    double * tmp_vec_X = NULL;
    double * tmp_mat_X = NULL;
    double * tmp_mat_XU = NULL;
    double * ck_msgf_m_Xnks = NULL;
    double * ck_msgf_V_Xnks = NULL;
    double * dual_marginal_Un = NULL;
    double * msgf_m_Xnk = NULL;
    double * msgf_V_Xnk = NULL;
    double * dual_marginal_Xn = NULL;
    double * gammas = NULL;
    double * msgb_xi_Ynks = NULL;
    double * msgb_W_Ynks = NULL;
    double * B_msgf_m_Uns = NULL;
    double * B_msgf_V_Uns_B_trans = NULL;
    double * msgf_V_Uns_B_trans = NULL;
    double * dual_marginal_Ynks = NULL;
 


    
    m_Xn= (double *) malloc(M*sizeof(double));
    memset(m_Xn,0,M*sizeof(double));


    msgf_V_Ynks= (double *) malloc(N*K*sizeof(double));
    memset(msgf_V_Ynks,0,N*K*sizeof(double));


    tmp_vec_X= (double *) malloc(M* sizeof(double));
    memset(tmp_vec_X,0,M*sizeof(double));


    tmp_mat_X= (double *) malloc(M*M* sizeof(double));
    memset(tmp_mat_X,0,M*M*sizeof(double));


    tmp_mat_XU= (double *) malloc(M*L* sizeof(double));
    memset(tmp_mat_XU,0,M*L*sizeof(double));

  
    ck_msgf_m_Xnks= (double *) malloc(N*K*sizeof(double));
    memset(ck_msgf_m_Xnks,0,N*K*sizeof(double));


    ck_msgf_V_Xnks= (double *) malloc(N*K*M*sizeof(double));
    memset(ck_msgf_V_Xnks,0,N*K*M*sizeof(double));


 
    dual_marginal_Un= (double *) malloc(L* sizeof(double));
    memset(dual_marginal_Un,0,L*sizeof(double));


    msgf_m_Xnk= (double *) malloc(M* sizeof(double));
    memset(msgf_m_Xnk,0,M*sizeof(double));

  
    msgf_V_Xnk= (double *) malloc(M*M* sizeof(double));
    memset(msgf_V_Xnk,0,M*M*sizeof(double));


    dual_marginal_Xn= (double *) malloc(M* sizeof(double));
    memset(dual_marginal_Xn,0,M*sizeof(double));






    msgb_xi_Ynks= (double *) malloc(N*K*sizeof(double));
    memset(msgb_xi_Ynks,0,N*K*sizeof(double));

    B_msgf_V_Uns_B_trans =(double *) malloc(M*M*sizeof(double));
    memset(B_msgf_V_Uns_B_trans,0,M*M*sizeof(double));
  
    msgf_V_Uns_B_trans =(double *) malloc(L*M*sizeof(double));
    memset(msgf_V_Uns_B_trans,0,L*M*sizeof(double));

    gammas= (double *) malloc(N*K*sizeof(double));
    memset(gammas,0,N*K*sizeof(double));

    msgb_W_Ynks= (double *) malloc(N*K*sizeof(double));
    memset(msgb_W_Ynks,0,N*K*sizeof(double));
    
    dual_marginal_Ynks = (double *) malloc(N*K*sizeof(double));
    memset(dual_marginal_Ynks,0,N*K*sizeof(double));

    B_msgf_m_Uns= (double *) malloc(M*sizeof(double));
    memset(B_msgf_m_Uns,0,M*sizeof(double));







    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,L,1.,B,L,msgf_m_Uns,1,0.,B_msgf_m_Uns,1);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,L,M,L,1.0,msgf_V_Uns,L,B,L,0.,tmp_mat_XU,M);
    cblas_dcopy(L*M,tmp_mat_XU,1,msgf_V_Uns_B_trans,1);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,M,L,1.0,B,L,msgf_V_Uns_B_trans,M,0.,B_msgf_V_Uns_B_trans,M);


    
    memset(objs,0,(maxit+1)*sizeof(double));
    obj_val = evaluate_quadratic_objective(M, m_X1,msgf_m_X1, msgf_W_X1);
    cblas_dcopy(M,m_X1,1,m_Xn,1);
    for(n=0;n<N;n++){
        obj_val = obj_val + evaluate_quadratic_objective(L, m_Us+n*L, msgf_m_Uns, msgf_W_Uns);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,K,M,1.,C,M,m_Xn,1,0.,m_Ys+n*K,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,m_Xn,1,0.,tmp_vec_X,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,L,1.,B,L,m_Us+n*L,1,1.,tmp_vec_X,1);
        cblas_dcopy(M,tmp_vec_X,1,m_Xn,1);
        for(k=0;k<K;k++)
        {   
            gammas[n*K+k] = init_gamma;
            nup_dual_unknown_parameters(choice, is_constrained,gammas[n*K+k], dual_marginal_Ynks[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k], msgb_xi_Ynks+n*K+k, msgb_W_Ynks+n*K+k);
            //nup_unknown_parameters(choice, m_Ys[n*K+k], betas[n*K+k], lower_bounds[n*K+k], upper_bounds[n*K+k], msgb_m_Ynks+n*K+k, msgb_V_Ynks+n*K+k);
            
            if(is_constrained<0){
                obj_val = obj_val + nup_loss_function(choice, m_Ys[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k]);
            }
        }
    }
    objs[0] = obj_val;


   it = 1;

    

    while(it<=maxit){

        obj_val = 0.;
        constraint_variation = 0.;

        // forward filtering
        cblas_dcopy(M,msgf_m_X1,1,msgf_m_Xnk,1);
        cblas_dcopy(M*M,msgf_V_X1,1,msgf_V_Xnk,1);
        for(n=0;n<N;n++){
            for(k=0;k<K;k++){
                ck_msgf_m_Xnks[n*K+k] = cblas_ddot(M,C+k*M,1,msgf_m_Xnk,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,M,M,1.,msgf_V_Xnk,M,C+k*M,1,0.,ck_msgf_V_Xnks+n*K*M+k*M,1);
                msgf_V_Ynks[n*K+k] = cblas_ddot(M,C+k*M,1,ck_msgf_V_Xnks+n*K*M+k*M,1);
                
                if(gammas[n*K+k]>1e10){
                    Gnk = 0.0;
                    gnk = -dual_marginal_Ynks[n*K+k];
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,gnk,msgf_V_Xnk,M,C+k*M,1,1.,msgf_m_Xnk,1);
                }else{
                    Gnk = msgb_W_Ynks[n*K+k]/(1.+msgb_W_Ynks[n*K+k]*msgf_V_Ynks[n*K+k]);
                    gnk = (msgb_xi_Ynks[n*K+k]-msgb_W_Ynks[n*K+k]*ck_msgf_m_Xnks[n*K+k])/(1.+msgb_W_Ynks[n*K+k]*msgf_V_Ynks[n*K+k]);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,gnk,msgf_V_Xnk,M,C+k*M,1,1.,msgf_m_Xnk,1);
                    cblas_dger(CblasRowMajor, M, M, -Gnk, ck_msgf_V_Xnks+n*K*M+k*M, 1, ck_msgf_V_Xnks+n*K*M+k*M, 1, msgf_V_Xnk, M);
                }
                
            }

            cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,msgf_m_Xnk,1,0.,tmp_vec_X,1);
            cblas_daxpy(M,1.,B_msgf_m_Uns,1,tmp_vec_X,1);
            cblas_dcopy(M,tmp_vec_X,1,msgf_m_Xnk,1);


            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,M,M,1.0,msgf_V_Xnk,M,A,M,0.,tmp_mat_X,M);
            cblas_dcopy(M*M,B_msgf_V_Uns_B_trans,1,msgf_V_Xnk,1);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,M,M,1.0,A,M,tmp_mat_X,M,1.,msgf_V_Xnk,M);
        }
        
        //backward dual deciding
        cblas_dcopy(M,dual_xi_Xf,1,dual_marginal_Xn,1);
        obj_val = 0.0;
        for(n=N-1;n>=0;n--){
            
            cblas_dcopy(L,msgf_m_Uns,1,m_Us+n*L,1);
            cblas_dgemv(CblasRowMajor,CblasNoTrans,L,M,-1.,msgf_V_Uns_B_trans,M,dual_marginal_Xn,1,1.,m_Us+n*L,1);
            obj_val = obj_val + evaluate_quadratic_objective(L, m_Us+n*L, msgf_m_Uns, msgf_W_Uns);
            cblas_dgemv(CblasRowMajor,CblasTrans,M,M,1.,A,M,dual_marginal_Xn,1,0.,tmp_vec_X,1);
            cblas_dcopy(M,tmp_vec_X,1,dual_marginal_Xn,1);
            for(k=K-1;k>=0;k--){
                    tmp_val = cblas_ddot(M,ck_msgf_V_Xnks+n*K*M+k*M,1,dual_marginal_Xn,1);
                    msgf_W_Ynk = 1.0/msgf_V_Ynks[n*K+k];
                    msgf_m_Ynk = ck_msgf_m_Xnks[n*K+k]-tmp_val;
                    msgf_xi_Ynk = msgf_W_Ynk*msgf_m_Ynk;

                    gamma_next = nup_dual_adaptive_parameter(choice, is_constrained, beta, msgf_xi_Ynk, msgf_W_Ynk, lower_bounds[n*K+k], upper_bounds[n*K+k]);
                    gammas[n*K+k] = fmax(gammas[n*K+k],gamma_next);



                    dual_marginal_Ynk = nup_dual_deciding(choice, is_constrained,msgf_xi_Ynk, msgf_W_Ynk, beta, lower_bounds[n*K+k], upper_bounds[n*K+k]);


                    m_Ynk = msgf_m_Ynk - msgf_V_Ynks[n*K+k]*dual_marginal_Ynk;

                    dual_marginal_Ynks[n*K+k] = dual_marginal_Ynk;
                    m_Ys[n*K+k] = m_Ynk;

                    
                    nup_dual_unknown_parameters(choice, is_constrained,gammas[n*K+k], dual_marginal_Ynks[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k], msgb_xi_Ynks+n*K+k, msgb_W_Ynks+n*K+k);



                cblas_daxpy(M,dual_marginal_Ynks[n*K+k],C+k*M,1,dual_marginal_Xn,1);
                if(is_constrained<0){
                    constraint_variation = constraint_variation + nup_loss_function(choice, m_Ys[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k]);
                    obj_val = obj_val + nup_loss_function(choice, m_Ys[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k]);
                }else{
                    constraint_variation = constraint_variation + nup_loss_function(choice, m_Ys[n*K+k], beta, lower_bounds[n*K+k], upper_bounds[n*K+k]);
                }
            }
        }
        
        cblas_dcopy(M,msgf_m_X1,1,m_X1,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,-1.,msgf_V_X1,M,dual_marginal_Xn,1,1.,m_X1,1);
        obj_val = obj_val + evaluate_quadratic_objective(M, m_X1,msgf_m_X1, msgf_W_X1);

        objs[it] = obj_val;


        if (it > 15){
            variation = 0.;
            for (i=0; i < 10; i++ ){
                variation += fabs(objs[it-1-i] - objs[it-i-2]);
                }
            if (variation < fabs(objs[it-1]) * 10 * obj_tol){
                printf("objective function converged at %d iteration\n",it-1);
                break;
            }
        }
        //printf("IFFBD: it = %d, cost function = %.12f, constraint variation = %f\n",it,objs[it],constraint_variation);
        it++;


        

    }
    
    

    free(m_Xn);
    free(msgf_V_Ynks);
    free(tmp_vec_X);
    free(tmp_mat_X);
    free(tmp_mat_XU);
    free(ck_msgf_m_Xnks);
    free(ck_msgf_V_Xnks);
    free(dual_marginal_Un);
    free(msgf_m_Xnk);
    free(msgf_V_Xnk);
    free(dual_marginal_Xn);
    free(gammas);
    free(msgb_xi_Ynks);
    free(msgb_W_Ynks);
    free(B_msgf_m_Uns); 
    free(B_msgf_V_Uns_B_trans); 
    free(msgf_V_Uns_B_trans);
    free(dual_marginal_Ynks);


    return it-1;


}












