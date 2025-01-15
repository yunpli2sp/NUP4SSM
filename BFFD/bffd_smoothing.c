
/* bffd_smoothing.c
 *
 * Copyright (C) 2024 Yun-Peng Li, Hans-Andrea Loeliger.
 *
 * Date: 2024-2-13
 * Author: Yunpeng Li (yunpli@isi.ee.ethz.ch)
 * Brief: Performing MAP estimation for linear non-Gaussian state space models using
 *        BFFD (backward filtering forward deciding) algorithm. The density of non-Gaussian
 *        inputs are assumed to be NUP (normal with unknown parameters) representations.
 *        function 'bffd_input_estimation' is designed for general MAP estimation in 
 *        non-Gaussian SSM;
 *        function 'bffd_constrained_control' is designed for linear quadratic control with
 *        halfspace/box constrained control signals. 
 * Cite: Y.-P Li, H.-A. Loeliger, “Backward Filtering Forward Deciding in Linear Non-Gaussian 
 *        State Space Models,” (to be appeared) in 2024 International Conference on Artificial 
 *        Intelligence and Statistics (AISTATS), May 2024.
 */





#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h> // MacOs
//#include <cblas.h> // Linux
#include "bffd_smoothing.h"
#include "routines.h"


int solve_equations(int n,double * A, double * b){
  
  int  info=0,nrhs=1; 
  //char uplo='L';
  int *ipiv = malloc((n+1)* sizeof(double));
  int lwork = n*n;
  double *work = malloc((lwork)* sizeof(double));
  char C  = 'N';

  memset(ipiv,0,(n+1)* sizeof(double));
  memset(work,0,(lwork)* sizeof(double));
  
  dgetrf_(&n,&n,A,&n,ipiv,&info);
  if(info!=0){
    //printf("\n failed to compute the LU decomposition: info =%d  \n",info);
    free(ipiv);
    free(work);
    return info;
  }
    
  dgetrs_(&C,&n,&nrhs,A,&n,ipiv,b,&n,&info);
  //if(info!=0) printf("\n failed to solve linear equations: info =%d  \n",info);

  free(ipiv);
  free(work);
  return info;

}

int initial_state(int M, double *msgf_Xi_X1, double *msgf_W_X1, double *msgb_Xi_X1, double *msgb_W_X1, double *sol)
{   
    int i = 0;
    int j = 0;
    int info = 0;
    double *A = malloc(M*M* sizeof(double));

   memset(sol,0,M*sizeof(double));

    for(i = 0;i < M; i++){
      sol[i] = msgf_Xi_X1[i] + msgb_Xi_X1[i];
      for(j = 0;j < M; j++){
        A[i*M+j] = msgf_W_X1[i*M+j] + msgb_W_X1[i*M+j];
      }
    }

    info = solve_equations(M,A,sol);
    return info;
}

/*
===========================================
choice                inputs' distribution
0                     Laplace/L1
1                     Hubber loss
2                     Hinge loss
3                     Vapnik loss
4                     Plain NUV
===========================================
*/
void NUP_unknown_parameters(int choice, double u, double beta, double *NUP_parameters, double *msgf_m_U, double *msgf_V_U)
{
  if(choice==1){
        laplace_unknown_parameters(u, beta, msgf_m_U, msgf_V_U);
      }else if(choice==2){
        huber_unknown_parameters(u, beta, NUP_parameters[0],msgf_m_U, msgf_V_U);
      }else if(choice==3){
        hinge_unknown_parameters(u, beta, NUP_parameters[0], msgf_m_U, msgf_V_U);
      }else if(choice==4){
        vapnik_unknown_parameters(u, beta, NUP_parameters[0], NUP_parameters[1], msgf_m_U, msgf_V_U);
      }else if(choice==5){
        plainNUV_unknown_parameters(u, msgf_m_U, msgf_V_U);
      }else{
        return;
      }
}
double NUP_decisions(int choice, double msgb_m_U, double msgb_V_U, double beta, double *NUP_parameters)
{
  if(choice==1){
    return laplace_decision(msgb_m_U, msgb_V_U, beta);
  }else if(choice==2){
    return huber_decision(msgb_m_U, msgb_V_U, beta, NUP_parameters[0]);
  }else if(choice==3){
    return hinge_decision(msgb_m_U, msgb_V_U, beta, NUP_parameters[0]);
  }else if(choice==4){
    return vapnik_decision(msgb_m_U, msgb_V_U, beta, NUP_parameters[0], NUP_parameters[1]);
  }else if(choice==5){
    return plainNUV_decision(msgb_m_U, msgb_V_U);
  }else{
    return msgb_m_U/(1.+beta*msgb_V_U);
  }
}
double NUP_regularizations(int choice, double m_U, double beta, double *NUP_parameters)
{
  if(choice==1){
    return laplace_regularization(m_U, beta);
  }else if(choice==2){
    return huber_regularization(m_U, beta, NUP_parameters[0]);
  }else if(choice==3){
    return hinge_regularization(m_U, beta, NUP_parameters[0]);
  }else if(choice==4){
    return vapnik_regularization(m_U, beta, NUP_parameters[0], NUP_parameters[1]);
  }else if(choice==5){
    return plainNUV_regularization(m_U);
  }else{
    return (beta/2.)*m_U*m_U;
  }
}



/**
 * C version of the BFFD smoothing for Linear Non-Gaussian SSM with NUP inputs
 * Considering the state space model
 * Y_{n} = CX_{n},                      
 * tilde{Y}_{n} = Y_{n} + Z_{n},       
 * X_{n+1} = AX_{n} + BU_{n},          
 * with non-Gaussian inputs U_{n}, and zero Gaussian (measurement) noise Z_{n}.
 * Given the observations y_obs, the covariance matrix msgb_W_Yn of tilde{Y}_{n},
 * as well as the choice of NUP priors imposed on inputs U_{n}, we want to obtain
 * the estimated m_Us of inputs U_{n} and the initial state X_{1}'s estimation m_X1, 
 * then we use them to reconstruct the pure ouputs $m_Ys$ for Y_{n} (n=1,...,N).
 * 
 * 
 * Arguments
 * ------
 * choice: int 
 *         the choice of NUP prior imposed on intputs U_{n}.
 * 
 *                  The selection of NUP priors and their parameters.
 *          =======================================================================
 *               choice         NUP prior          NUP_parameters[0],[1]
 *          =======================================================================    
 *                0            Laplace/L1               /,/
 *                1            Hubber loss          r (threshold),/
 *                2            Hinge loss           a (lower bound),/
 *                3            Vapnik loss       a (lower bound),b (upper bound)
 *                4            Plain NUV               /,/ 
 *          =======================================================================
 * 
 * NUP_parameters: double array of size 2 
 *                 the parameters of selected NUP prior.
 * beta: double 
 *       the non-negative scale parameter for NUP priors.
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
 * max_valid_index: int
 *                  when the final state is unknown,it is the maxmum index 
 *                  of inputs which contributes to the y_obs. default value
 *                  is N-1.
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
 * y_obs: double array of size N*K
 *        the values of output observations.
 * 
 * msgf_m_X1: double array of size M
 *             the (forward) mean of initial state X_{1}, default is 
 *             zero vector.
 *  
 * msgf_W_X1: double array of size M*M
 *            the (forward) precision matrix of initial state X_{1}, default is
 *            zero matrix.
 * 
 * msgf_m_Xf: double array of size M
 *             the (backward) mean of final state X_{N+1}, default is 
 *             zero vector.
 * 
 * msgf_W_Xf: double array of size M*M
 *            the (backward) precision matrix of final state X_{N+1}, default is
 *            zero matrix.
 * 
 * msgb_W_Yn: double array of size K*K
 *            the precision matrix of Gaussian noise Z_{n} (or tilde{Y_{n}}).
 * 
 * maxit: int
 *        maximum number of iterations in optimization.
 * 
 * obj_tol: double 
 *          stopping criteria tolerance
 * 
 * eps: double
 *      small value used to prevent zero-division
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

int bffd_input_estimation(int choice, double *NUP_parameters, double beta, int N, int M, int L, int K, int max_valid_index,double *A, double *B, 
                          double *C, double *y_obs,double *msgf_m_X1, double *msgf_W_X1, double *msgb_m_Xf, double *msgb_W_Xf,double *msgb_W_Yn,
                          int maxit, double obj_tol,double eps, double *m_X1, double *m_Us, double *m_Ys, double *objs)
{   
    int n = 0;
    int l = 0;   
    int it = 0;
    int i = 0;
    int info = 0;
    int max_index = N;

    double msgb_m_U_element = 0.0;
    double msgb_V_U_element = 0.0;
    double H = 0;
    
    

    double obj_ls = 0.0;
    double obj_plt = 0.0;
    double tmp_val = 0.0;
    double tmp_val2 = 0.0;

    


    double variation = 0.;

    double *msgb_Xi_Yn;

    double* b;
    double* h;

    double* m_X;
    double* V_X;
    double* m_Y;
    double* msgb_Xi_Xs;
    double* msgb_W_Xs;
    double* msgf_m_Us;
    double* msgf_V_Us;


    double* tmp_mat;
    double* tmp_mat2;
    double* precomputed_vectors;
    double* precomputed_matrix; 

    double* tmp_vec;
    double* tmp_vec_Y;

    double *msgf_Xi_X1;
    double *msgb_Xi_Xf;

  
    

    if(choice==1){
      printf("NUP prior : Laplace/L1 \n");
    }else if(choice==2){
      printf("NUP prior : Huber loss \n");
    }else if(choice==3){
      printf("NUP prior : hinge loss \n");
    }else if(choice==4){
      printf("NUP prior : Vapnik loss \n");
    }else if(choice==5){
      printf("NUP prior : plain NUV \n");
    }else{
      printf("Error: illegal value of 'choice' \n");
      return -1;
    }
     

  


 
    memset(objs,0,maxit*sizeof(double));

    msgb_Xi_Yn = (double *) malloc(K* sizeof(double));
    memset(msgb_Xi_Yn,0,K*sizeof(double));

    b= (double *) malloc(M* sizeof(double));
    memset(b,0,M*sizeof(double));

    h= (double *) malloc(M* sizeof(double));
    memset(h,0,M*sizeof(double));

    m_X = (double *) malloc(M* sizeof(double));
    memset(m_X,0,M*sizeof(double));

    V_X = (double *) malloc(M*M* sizeof(double));
    memset(V_X,0,M*M*sizeof(double));

    m_Y = (double *) malloc(K* sizeof(double));
    memset(m_Y,0,K*sizeof(double));

    msgb_Xi_Xs = (double *) malloc((N+1)*(L+1)*M* sizeof(double));
    memset(msgb_Xi_Xs,0,(N+1)*(L+1)*M* sizeof(double));

    msgb_W_Xs = (double *) malloc((N+1)*(L+1)*M*M* sizeof(double));
    memset(msgb_W_Xs,0,(N+1)*(L+1)*M*M* sizeof(double));

   
    msgf_m_Us = (double *) malloc(N*L*sizeof(double));
    memset(msgf_m_Us,0,N*L*sizeof(double));

    msgf_V_Us = (double *) malloc(N*L*sizeof(double));
    memset(msgf_V_Us,0,N*L*sizeof(double));


    tmp_mat = (double *) malloc(K*M*sizeof(double));
    memset(tmp_mat,0,K*M*sizeof(double));

    tmp_mat2 = (double *) malloc(M*M*sizeof(double));
    memset(tmp_mat2,0,M*M*sizeof(double));

    precomputed_vectors = (double *) malloc(N*M*sizeof(double));
    memset(precomputed_vectors,0,N*M*sizeof(double));

    precomputed_matrix = (double *) malloc(M*M*sizeof(double));
    memset(precomputed_matrix,0,M*M*sizeof(double));


    tmp_vec = (double *) malloc(M*sizeof(double));
    memset(tmp_vec,0,M*sizeof(double));

    tmp_vec_Y = (double *) malloc(K*sizeof(double));
    memset(tmp_vec_Y,0,K*sizeof(double));

    msgf_Xi_X1 = (double *) malloc(M*sizeof(double));
    memset(msgf_Xi_X1,0,M*sizeof(double));


    msgb_Xi_Xf = (double *) malloc(M*sizeof(double));
    memset(msgb_Xi_Xf,0,M*sizeof(double));


    /*
    initialization
    */

    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgf_W_X1,M,msgf_m_X1,1,0,msgf_Xi_X1,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xf,M,msgb_m_Xf,1,0,msgb_Xi_Xf,1);
    
    for(n=0;n<N;n++){
      cblas_dgemv(CblasRowMajor,CblasNoTrans,K,K,1.,msgb_W_Yn,K,y_obs+n*K,1,0,tmp_vec,1);
      cblas_dgemv(CblasRowMajor,CblasTrans,K,M,1.0,C,M,tmp_vec,1,0.,precomputed_vectors+n*M,1);
      for(l=0;l<L;l++){
        msgf_m_Us[n*L+l] = 0.0;
        msgf_V_Us[n*L+l] = 1.0/beta;
      }
    }
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,K,M,K,1.0,msgb_W_Yn,K,C,M,0.,tmp_mat,M);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,M,M,K,1.0,C,M,tmp_mat,M,0.,precomputed_matrix,M); 

    tmp_val = 0.;
    for(i=0;i<M;i++){
      tmp_val = tmp_val + cblas_ddot(M,msgb_W_Xf+i*M,1,msgb_W_Xf+i*M,1);
    }

    if(tmp_val<1e-16){
      max_index = max_valid_index;
      //cblas_dcopy(M,precomputed_vectors+(N-1)*M,1,msgb_Xi_Xs+(N-1)*(L+1)*M+L*M,1);
      //cblas_dcopy(M*M,precomputed_matrix,1,msgb_W_Xs+(N-1)*(L+1)*M*M + L*M*M,1);
    }
    else{
      max_index = N;
      cblas_dcopy(M,msgb_Xi_Xf,1,msgb_Xi_Xs+N*(L+1)*M+L*M,1);
      cblas_dcopy(M*M,msgb_W_Xf,1,msgb_W_Xs+N*(L+1)*M*M + L*M*M,1);
    }



    
    

    while(it<(maxit-1)){

    obj_ls = 0.;
    obj_plt = 0.;

    // backward filtering
    for(n=N-1;n>=0;n--){

      for(l=L;l>=1;l--){

        cblas_dcopy(M,B+l-1,L,b,1);

        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,b,1,0,tmp_vec,1);
        tmp_val = cblas_ddot(M,b,1,tmp_vec,1);
        tmp_val = 1. + msgf_V_Us[n*L+l-1]*tmp_val;
        H = msgf_V_Us[n*L+l-1]/tmp_val;

        tmp_val2 = msgf_m_Us[n*L+l-1] + msgf_V_Us[n*L+l-1]*cblas_ddot(M,b,1,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1);
        memset(h,0,M* sizeof(double));
        cblas_daxpy(M,tmp_val2/tmp_val,b,1,h,1);

        cblas_dcopy(M,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1,msgb_Xi_Xs+(n+1)*(L+1)*M+(l-1)*M,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,-1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,h,1,1.,msgb_Xi_Xs+(n+1)*(L+1)*M+(l-1)*M,1);

        cblas_dcopy(M*M,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,1,msgb_W_Xs+(n+1)*(L+1)*M*M + (l-1)*M*M,1);
        cblas_dger(CblasRowMajor, M, M, 0.- H, tmp_vec, 1, tmp_vec, 1, msgb_W_Xs+(n+1)*(L+1)*M*M + (l-1)*M*M, M);
        
      }


      
      cblas_dcopy(M,precomputed_vectors+n*M,1,msgb_Xi_Xs+n*(L+1)*M+L*M,1);
      cblas_dgemv(CblasRowMajor,CblasTrans,M,M,1.,A,M,msgb_Xi_Xs+(n+1)*(L+1)*M,1,1.,msgb_Xi_Xs+n*(L+1)*M+L*M,1);
 
      cblas_dcopy(M*M,precomputed_matrix,1,msgb_W_Xs+n*(L+1)*M*M + L*M*M,1);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,M,M,1.0,msgb_W_Xs+(n+1)*(L+1)*M*M,M,A,M,0.,tmp_mat2,M);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,M,M,M,1.0,A,M,tmp_mat2,M,1.,msgb_W_Xs+n*(L+1)*M*M + L*M*M,M); 


    }
    // computing the estimation of initial state X_{1}.
    info = initial_state(M, msgf_Xi_X1, msgf_W_X1, msgb_Xi_Xs+L*M, msgb_W_Xs+L*M*M,m_X1);
    cblas_dcopy(M,m_X1,1,m_X,1);

   
    cblas_daxpy(M,-1.,msgf_m_X1,1,m_X,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgf_W_X1,M,m_X,1,0,tmp_vec,1);
    //obj_ls = obj_ls + cblas_ddot(M,m_X,1,tmp_vec,1);
    objs[it+1] = evaluate_least_squares(M, m_X1, msgf_m_X1, msgf_W_X1);

    cblas_dcopy(M,m_X1,1,m_X,1);

    


    // forward deciding
    for(n=0;n<N;n++){
      

      cblas_dgemv(CblasRowMajor,CblasNoTrans,K,M,1.,C,M,m_X,1,0,m_Y,1);
      cblas_dcopy(K,m_Y,1,m_Ys+n*K,1);
      cblas_daxpy(K,-1.,y_obs+n*K,1,m_Y,1);
      cblas_dgemv(CblasRowMajor,CblasNoTrans,K,K,1.,msgb_W_Yn,K,m_Y,1,0,tmp_vec_Y,1);
      //obj_ls = obj_ls + cblas_ddot(K,m_Y,1,tmp_vec_Y,1);
      objs[it+1] = objs[it+1] + evaluate_least_squares(K, m_Ys+n*K, y_obs+n*K, msgb_W_Yn);
      
      cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,m_X,1,0,tmp_vec,1);
      cblas_dcopy(M,tmp_vec,1,m_X,1);


      for(l=1;l<=L;l++){

        cblas_dcopy(M,B+l-1,L,b,1);

        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,b,1,0,tmp_vec,1);
        //msgb_V_U_element = 1.0/cblas_ddot(M,b,1,tmp_vec,1);




        if(n<max_index){
           msgb_V_U_element = 1.0/(cblas_ddot(M,b,1,tmp_vec,1));  // if the results are overflow, using this for numerical stability.
           tmp_val = cblas_ddot(M,b,1,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1) - cblas_ddot(M,tmp_vec,1,m_X,1);
           msgb_m_U_element = msgb_V_U_element*tmp_val;
           m_Us[n*L+l-1] =  NUP_decisions(choice, msgb_m_U_element, msgb_V_U_element , beta, NUP_parameters);
           NUP_unknown_parameters(choice, m_Us[n*L+l-1], beta, NUP_parameters, msgf_m_Us+n*L+l-1, msgf_V_Us+n*L+l-1);
        }else{
          //m_Us[n*L+l-1] = map_estimation(msgf_m_Us[n*L+l-1],msgf_V_Us[n*L+l-1],msgb_m_U_element,msgb_V_U_element);
          m_Us[n*L+l-1] = 0.0;
        }
        //obj_plt = obj_plt + NUP_regularizations(choice, m_Us[n*L+l-1], beta, NUP_parameters);
        objs[it+1] = objs[it+1] + NUP_regularizations(choice, m_Us[n*L+l-1], beta, NUP_parameters);




        cblas_daxpy(M,m_Us[n*L+l-1],b,1,m_X,1);


      }
    }

    cblas_daxpy(M,-1.,msgb_m_Xf,1,m_X,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xf,M,m_X,1,0,tmp_vec,1);
    //obj_ls = obj_ls + cblas_ddot(M,m_X,1,tmp_vec,1);

    
    //objs[it] = 0.5*obj_ls + obj_plt;




     
    
    if (it > 20)
    {
      variation = 0.;
      for (i=0; i < 10; i++ ){
        variation += fabs(objs[it-1-i] - objs[it-i-2]);
      }
      
     
      if (variation < fabs(objs[it-1]) * 10 * obj_tol){
          printf("objective function converged at %d iteration\n",it-1);
          break;
      }
    }


      it++;
    }



    free(msgb_Xi_Yn);

    free(b);
    free(h);

    free(m_X);
    free(V_X);
    free(m_Y);
    free(msgb_Xi_Xs);
    free(msgb_W_Xs);
    free(msgf_m_Us);
    free(msgf_V_Us);


    free(tmp_mat);
    free(tmp_mat2);
    free(precomputed_vectors);
    free(precomputed_matrix); 

    free(tmp_vec);
    free(tmp_vec_Y);
    free(msgf_Xi_X1);
    free(msgb_Xi_Xf);





    return it;
}

/**
 * C version of the BFFD smoothing for Linear quadratic constrained control
 * Considering the state space model      
 * X_{n+1} = AX_{n} + BU_{n},          
 * driven by linear constrained inputs U_{n}.
 * Given the initial state x1, final state xf, state cost matrix Q, final state cost matrix Qf,
 * time horizon N, as well as the choice of linear constraints imposed on inputs U_{n}, we want 
 * to steer the state X_{n} from initial state x1 to final state xf using inputs U_{n} (n=1,...,N).
 * 
 * Arguments
 * ------
 * choice: int 
 *         the choice of linear constraints imposed on intputs U_{n}.
 *              The selection of NUP priors and their parameters.
 *          =======================================================================
 *              choice         NUP prior          NUP_parameters[0],[1]
 *                             (constraint)
 *          =======================================================================    
 *                2            Hinge loss        a (lower bound),/
 *                             (halfspace)
 *                3            Vapnik loss       a (lower bound),b (upper bound)
 *                             (box)
 *          =======================================================================
    
* NUP_parameters: double array of size 2 
*                 the parameters of selected NUP prior.

* N: double
*    the value of sampe size.
* 
* M: double
*    the dimension of states X_{n}.
* 
* L: double
*    the dimension of inputs U_{n}.
* 
* A: double array of size M*M
*    the state-transition matrix.
* 
* B: double array of size M*L
*    the input matrix.
* 
* Q: double array of size M*M
*    the state cost matrix.
*
* Qf: double array of size M*M
*     the final state cost matrix.
* 
* x1: double array of size M
*     the value of initial state.
*
* xf: double array of size M
*     the value of final state.
*  
* maxit: int
*        maximum number of iterations in optimization.
* 
* obj_tol: double 
*          stopping criteria tolerance
* 
* eps: double
*      small value used to prevent zero-division
* 
* m_Xs: double array of size (N+1)*M
*       the estimation of SSM states X_{n}.
*
* m_Us: double array of size N*L
*       the estimation of SSM inputs U_{n}.
* 
* objs: double array of size maxit
*       the value of cost fuction along the iteration.
* 
* Return
* -------
* total_iter: int
*             the total number of iterations consumed during optimization.
*/

int bffd_constrained_control(int choice, double * NUP_parameters, int N, int M, int L, double * A, double * B,
double *Q, double *Qf, double * x1, double *xf,int maxit, double obj_tol, double eps, double *m_Xs,double *m_Us,double *objs){
    

    int n = 0;
    int l = 0;   
    int it = 0;
    int i = 0;
  

    double msgb_m_U_element = 0.0;
    double msgb_V_U_element = 0.0;
    double H = 0;
    
    

    double obj_ls = 0.0;
    double tmp_val = 0.0;
    double tmp_val2 = 0.0;

    


    double variation = 0.;
    double beta_new = 0.;

    double * betas;
    double * X_diff;



    double* b;
    double* h;

    double* m_X;
    double* V_X;
    double* msgb_Xi_Xs;
    double* msgb_W_Xs;
    double* msgf_m_Us;
    double* msgf_V_Us;

  
    double* tmp_mat2;
    double* precomputed_vectors;


    double* tmp_vec;


    


    if(choice==2){
      printf("Enforcing halfspace constraints using hinge loss \n");
    }else if(choice==3){
      printf("Enforcing box constraints using Vapnik loss \n");
    }else{
      printf("Error: illegal value of 'choice' \n");
      return -1;
    }

  



    memset(objs,0,maxit*sizeof(double));




    betas = (double *) malloc(N*L* sizeof(double));
    memset(betas,0,N*L*sizeof(double));

    X_diff = (double *) malloc(M*sizeof(double));
    memset(X_diff,0,M*sizeof(double));



    b= (double *) malloc(M* sizeof(double));
    memset(b,0,M*sizeof(double));

    h= (double *) malloc(M* sizeof(double));
    memset(h,0,M*sizeof(double));

    m_X = (double *) malloc(M* sizeof(double));
    memset(m_X,0,M*sizeof(double));

    V_X = (double *) malloc(M*M* sizeof(double));
    memset(V_X,0,M*M*sizeof(double));



    msgb_Xi_Xs = (double *) malloc((N+1)*(L+1)*M* sizeof(double));
    memset(msgb_Xi_Xs,0,(N+1)*(L+1)*M* sizeof(double));

    msgb_W_Xs = (double *) malloc((N+1)*(L+1)*M*M* sizeof(double));
    memset(msgb_W_Xs,0,(N+1)*(L+1)*M*M* sizeof(double));

   
    msgf_m_Us = (double *) malloc(N*L*sizeof(double));
    memset(msgf_m_Us,0,(N+1)*L*sizeof(double));

    msgf_V_Us = (double *) malloc(N*L*sizeof(double));
    memset(msgf_V_Us,0,(N+1)*L*sizeof(double));



    tmp_mat2 = (double *) malloc(M*M*sizeof(double));
    memset(tmp_mat2,0,M*M*sizeof(double));

    precomputed_vectors = (double *) malloc(2*M*sizeof(double));
    memset(precomputed_vectors,0,2*M*sizeof(double));





    tmp_vec = (double *) malloc(M*sizeof(double));
    memset(tmp_vec,0,M*sizeof(double));





 /*
    initialization
  */

    
    for(n=0;n<N;n++){
      for(l=0;l<L;l++){
        msgf_m_Us[n*L+l] = 0.0;
        msgf_V_Us[n*L+l] = 1.0;
        betas[n*L+l] = 5.;
      }
    }

    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,Q,M,xf,1,0,precomputed_vectors,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,Qf,M,xf,1,0,precomputed_vectors+M,1);

    cblas_dcopy(M,precomputed_vectors+M,1,msgb_Xi_Xs+N*(L+1)*M+L*M,1);
    cblas_dcopy(M*M,Qf,1,msgb_W_Xs+N*(L+1)*M*M + L*M*M,1);
    
    

    while(it<maxit){

    obj_ls = 0.;
    //backward filtering
    for(n=N-1;n>=0;n--){

      for(l=L;l>=1;l--){

        cblas_dcopy(M,B+l-1,L,b,1);

        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,b,1,0,tmp_vec,1);
        tmp_val = cblas_ddot(M,b,1,tmp_vec,1);
        tmp_val = 1. + msgf_V_Us[n*L+l-1]*tmp_val;
        H = msgf_V_Us[n*L+l-1]/tmp_val;

        tmp_val2 = msgf_m_Us[n*L+l-1] + msgf_V_Us[n*L+l-1]*cblas_ddot(M,b,1,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1);
        memset(h,0,M* sizeof(double));
        cblas_daxpy(M,tmp_val2/tmp_val,b,1,h,1);

        cblas_dcopy(M,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1,msgb_Xi_Xs+(n+1)*(L+1)*M+(l-1)*M,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,-1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,h,1,1.,msgb_Xi_Xs+(n+1)*(L+1)*M+(l-1)*M,1);

        cblas_dcopy(M*M,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,1,msgb_W_Xs+(n+1)*(L+1)*M*M + (l-1)*M*M,1);
        cblas_dger(CblasRowMajor, M, M, 0.- H, tmp_vec, 1, tmp_vec, 1, msgb_W_Xs+(n+1)*(L+1)*M*M + (l-1)*M*M, M);
        
      }

      
      cblas_dcopy(M,precomputed_vectors,1,msgb_Xi_Xs+n*(L+1)*M+L*M,1);
      cblas_dcopy(M*M,Q,1,msgb_W_Xs+n*(L+1)*M*M + L*M*M,1);

      cblas_dgemv(CblasRowMajor,CblasTrans,M,M,1.,A,M,msgb_Xi_Xs+(n+1)*(L+1)*M,1,1.,msgb_Xi_Xs+n*(L+1)*M+L*M,1);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,M,M,1.0,msgb_W_Xs+(n+1)*(L+1)*M*M,M,A,M,0.,tmp_mat2,M);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,M,M,M,1.0,A,M,tmp_mat2,M,1.,msgb_W_Xs+n*(L+1)*M*M + L*M*M,M); 
      
    }


    cblas_dcopy(M,x1,1,m_X,1);
    //forward deciding
    for(n=0;n<N;n++){
      
      cblas_dcopy(M,m_X,1,m_Xs+n*M,1);

      cblas_dcopy(M,xf,1,X_diff,1);
      cblas_daxpy(M,-1.,m_X,1,X_diff,1);
      cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,Q,M,X_diff,1,0,tmp_vec,1);
      obj_ls = obj_ls + cblas_ddot(M,X_diff,1,tmp_vec,1);

      memset(tmp_vec,0,M* sizeof(double));
      cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,m_X,1,0,tmp_vec,1);
      cblas_dcopy(M,tmp_vec,1,m_X,1);


      for(l=1;l<=L;l++){

        cblas_dcopy(M,B+l-1,L,b,1);

        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgb_W_Xs+(n+1)*(L+1)*M*M + l*M*M,M,b,1,0,tmp_vec,1);
        //msgb_V_U_element = 1.0/cblas_ddot(M,b,1,tmp_vec,1);
        msgb_V_U_element = 1.0/(eps + cblas_ddot(M,b,1,tmp_vec,1));

        tmp_val = cblas_ddot(M,b,1,msgb_Xi_Xs+(n+1)*(L+1)*M+l*M,1) - cblas_ddot(M,tmp_vec,1,m_X,1);
        msgb_m_U_element = msgb_V_U_element*tmp_val;

        m_Us[n*L+l-1] =  NUP_decisions(choice, msgb_m_U_element, msgb_V_U_element , betas[n*L+l-1], NUP_parameters);
        
        // adjust the scale parameters for constraint satisfication when necessary.
        if(choice==3){
          beta_new = hinge_adaptive_parameter(m_Us[n*L+l-1], msgb_m_U_element, msgb_V_U_element, NUP_parameters[0]);
        }else if(choice==4){
          beta_new = vapnik_adaptive_parameter(m_Us[n*L+l-1], msgb_m_U_element, msgb_V_U_element, NUP_parameters[0],NUP_parameters[1]);
        }

        if(beta_new>betas[n*L+l-1]){
            betas[n*L+l-1] = beta_new;
            m_Us[n*L+l-1] =  NUP_decisions(choice, msgb_m_U_element, msgb_V_U_element , betas[n*L+l-1], NUP_parameters);
        }

        NUP_unknown_parameters(choice, m_Us[n*L+l-1], betas[n*L+l-1], NUP_parameters, msgf_m_Us+n*L+l-1, msgf_V_Us+n*L+l-1);

        cblas_daxpy(M,m_Us[n*L+l-1],b,1,m_X,1);

      }
    }
    cblas_dcopy(M,m_X,1,m_Xs+N*M,1);

    cblas_dcopy(M,xf,1,X_diff,1);
    cblas_daxpy(M,-1.,m_X,1,X_diff,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,Qf,M,X_diff,1,0,tmp_vec,1);  
    obj_ls = obj_ls + cblas_ddot(M,X_diff,1,tmp_vec,1);


    
    objs[it] = obj_ls;
     

    if (it > 20)
    {
      variation = 0.;
      for (i=0; i < 10; i++ ){
        variation += fabs(objs[it-1-i] - objs[it-i-2]);
      }
      
     
      if (variation < fabs(objs[it-1]) * 10 * obj_tol){
          printf("objective function converged at %d iteration\n",it-1);
          break;
      }
    }


      it++;
    }







    free(betas);
    free(X_diff);

    free(b);
    free(h);

    free(m_X);
    free(V_X);
    free(msgb_Xi_Xs);
    free(msgb_W_Xs);
    free(msgf_m_Us);
    free(msgf_V_Us);


 
    free(tmp_mat2);
    free(precomputed_vectors);


    free(tmp_vec);


    return it-1;
}



