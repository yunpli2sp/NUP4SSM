#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h> // MacOs
//#include <cblas.h> // Linux
#include "routines.h"



double evaluate_least_squares(int M, double * m_X, double * mean_X, double * W_X)
{   double val = 0.0;
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

/* double evaluate_least_squares(int M, double * m_X, double * mean_X, double * W_X)
{   double val = 0.0;
    double * vec_1;
    double * vec_2;

    vec_1= (double *) malloc(M* sizeof(double));
    memset(vec_1,0,M*sizeof(double));

    vec_2= (double *) malloc(M* sizeof(double));
    memset(vec_2,0,M*sizeof(double));

    cblas_dcopy(M,m_X,1,vec_1,1);
    cblas_daxpy(M,-1.,mean_X,1,vec_1,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,W_X,M,vec_1,1,0,vec_2,1);
    val = 0.5*cblas_ddot(M,vec_1,1,vec_2,1);

    free(vec_1);
    free(vec_2);

    return val;
} */

double FE(int N, int M, int L, int K, double alpha, double *m_X1, double *m_Uns, double *p_X1, double *p_Uns,double *A, 
         double *B, double *C, double *msgf_m_X1, double *msgf_W_X1, double *msgf_m_Uns, double *msgf_W_Uns, double *y_obs, 
         double *msgb_W_Yns, double *m_Yns)
{   int n = 0;
    double phi = 0.0;
    double * m_Xn;
    double * tmp_vec_X;

    
    m_Xn= (double *) malloc(M* sizeof(double));
    memset(m_Xn,0,M*sizeof(double));

    tmp_vec_X= (double *) malloc(M* sizeof(double));
    memset(tmp_vec_X,0,M*sizeof(double));

    cblas_daxpy(M,alpha,p_X1,1,m_X1,1);
    
    phi = evaluate_least_squares(M, m_X1, msgf_m_X1, msgf_W_X1);
    cblas_dcopy(M,m_X1,1,m_Xn,1);



    for(n = 0; n<N; n++){
        cblas_daxpy(L,alpha,p_Uns+n*L,1,m_Uns+n*L,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,K,M,1.,C,M,m_Xn,1,0,m_Yns+n*K,1);
        phi = phi + evaluate_least_squares(K, m_Yns+n*K, y_obs+n*K, msgb_W_Yns);
        phi = phi + evaluate_least_squares(L, m_Uns+n*L, msgf_m_Uns, msgf_W_Uns);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,m_Xn,1,0,tmp_vec_X,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,L,1.,B,L,m_Uns+n*L,1,1.,tmp_vec_X,1);
        cblas_dcopy(M,tmp_vec_X,1,m_Xn,1);
    }   

    free(m_Xn);
    free(tmp_vec_X);
    return phi;
}

void BD(int N, int M, int L, int K, double *m_X1, double *m_Uns, double *m_Yns, double *A, double *B, double *C, 
double *msgf_m_X1, double *msgf_W_X1, double *msgf_m_Uns, double *msgf_W_Uns, double *y_obs, double *msgb_W_Yns, 
double * grad_X1, double * grad_Uns)
{
    int n = 0;
    double * msgb_S_Xn;
    double * tmp_vec_X;
    double * tmp_vec_U;
    double * tmp_vec_Y;
    double * tmp_vec_Y2;

    msgb_S_Xn= (double *) malloc(M* sizeof(double));
    memset(msgb_S_Xn,0,M*sizeof(double));

    tmp_vec_X= (double *) malloc(M* sizeof(double));
    memset(tmp_vec_X,0,M*sizeof(double));

    tmp_vec_U= (double *) malloc(L* sizeof(double));
    memset(tmp_vec_U,0,L*sizeof(double));

    tmp_vec_Y= (double *) malloc(K* sizeof(double));
    memset(tmp_vec_Y,0,K*sizeof(double));

    tmp_vec_Y2= (double *) malloc(K* sizeof(double));
    memset(tmp_vec_Y2,0,K*sizeof(double));
    
    for(n=N-1;n>=0;n--){
        cblas_dcopy(L,m_Uns+n*L,1,tmp_vec_U,1);
        cblas_daxpy(L,-1.,msgf_m_Uns,1,tmp_vec_U,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,L,L,1.,msgf_W_Uns,L,tmp_vec_U,1,0,grad_Uns+n*L,1);
        cblas_dgemv(CblasRowMajor,CblasTrans,M,L,-1.0,B,L,msgb_S_Xn,1,1.,grad_Uns+n*L,1);

        cblas_dgemv(CblasRowMajor,CblasTrans,M,M,1.0,A,M,msgb_S_Xn,1,0,tmp_vec_X,1);
        cblas_dcopy(K,y_obs+n*K,1,tmp_vec_Y,1);
        cblas_daxpy(K,-1.,m_Yns+n*K,1,tmp_vec_Y,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,K,K,1.,msgb_W_Yns,K,tmp_vec_Y,1,0,tmp_vec_Y2,1);
        cblas_dgemv(CblasRowMajor,CblasTrans,K,M,1.0,C,M,tmp_vec_Y2,1,1.,tmp_vec_X,1);
        cblas_dcopy(M,tmp_vec_X,1,msgb_S_Xn,1);
    }


    cblas_dcopy(M,m_X1,1,tmp_vec_X,1);
    cblas_dcopy(M,msgb_S_Xn,1,grad_X1,1);
    cblas_daxpy(M,-1.,msgf_m_X1,1,tmp_vec_X,1);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,msgf_W_X1,M,tmp_vec_X,1,-1.,grad_X1,1);

    


    
    free(msgb_S_Xn);
    free(tmp_vec_X);
    free(tmp_vec_U);
    free(tmp_vec_Y);
    free(tmp_vec_Y2);

    return;


}

double QI(int N, int M, int L, int K, double phi0, double phi0_der_1st, double alpha1, double *m_X1, double *m_Uns, 
          double *p_X1, double *p_Uns, double *m_Yns, double *A, double *B, double *C, double *msgf_m_X1, double *msgf_W_X1, 
          double *msgf_m_Uns, double *msgf_W_Uns, double *y_obs, double *msgb_W_Yns)
{   


    int n = 0;
    double alpha = 0.;
    double phi1 = 0.0;
    double * interp1_m_Xn;
    double * tmp_vec_X;
    double * interp1_m_Un;
    double * interp1_m_Yn;


    
    interp1_m_Xn= (double *) malloc(M* sizeof(double));
    memset(interp1_m_Xn,0,M*sizeof(double));

    tmp_vec_X= (double *) malloc(M* sizeof(double));
    memset(tmp_vec_X,0,M*sizeof(double));

    interp1_m_Un= (double *) malloc(L* sizeof(double));
    memset(interp1_m_Un,0,L*sizeof(double));

    interp1_m_Yn= (double *) malloc(K* sizeof(double));
    memset(interp1_m_Yn,0,K*sizeof(double));



    cblas_dcopy(M,m_X1,1,interp1_m_Xn,1);
    cblas_daxpy(M,alpha1,p_X1,1,interp1_m_Xn,1);
    
    phi1 = evaluate_least_squares(M, interp1_m_Xn, msgf_m_X1, msgf_W_X1);


    for(n = 0; n<N; n++){
        cblas_dcopy(L,m_Uns+n*L,1,interp1_m_Un,1);
        cblas_daxpy(L,alpha1,p_Uns+n*L,1,interp1_m_Un,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,K,M,1.,C,M,interp1_m_Xn,1,0,interp1_m_Yn,1);
        phi1 = phi1 + evaluate_least_squares(K, interp1_m_Yn, y_obs+n*K, msgb_W_Yns);
        phi1 = phi1 + evaluate_least_squares(L, interp1_m_Un, msgf_m_Uns, msgf_W_Uns);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,M,1.,A,M,interp1_m_Xn,1,0,tmp_vec_X,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,M,L,1.,B,L,interp1_m_Un,1,1.,tmp_vec_X,1);
        cblas_dcopy(M,tmp_vec_X,1,interp1_m_Xn,1);
    }   
    
    alpha = -(phi0_der_1st*alpha1*alpha1)/(2*(phi1-phi0-phi0_der_1st*alpha1));

    free(interp1_m_Xn);
    free(tmp_vec_X);
    free(interp1_m_Un);
    free(interp1_m_Yn);

    return alpha;

}

double maximum_scale_parameter(int N, int M, int L, int K, double * m_X1, double * m_Uns, double * m_Yns, double *p_X1, 
       double *p_Uns, double * A, double * B, double * C, double * msgf_m_X1, double * msgf_W_X1, double * y_obs, 
       double * msgb_W_Yns, double * grad_X1, double * grad_Uns){


        int max_index1 = 0;
        int max_index2 = 0;
        double phi = 0.0;
        double gamma_max = 0.0;

        double * msgf_m_Uns;
        double * msgf_W_Uns;

        msgf_m_Uns= (double *) malloc(L* sizeof(double));
        memset(msgf_m_Uns,0,L*sizeof(double));

        msgf_W_Uns= (double *) malloc(L*L* sizeof(double));
        memset(msgf_W_Uns,0,L*L*sizeof(double));
        
        memset(m_X1,0,M*sizeof(double));
        memset(m_Uns,0,N*L*sizeof(double));

        phi = FE(N, M, L, K, 0., m_X1, m_Uns, p_X1, p_Uns, A, B, C, msgf_m_X1, 
                 msgf_W_X1, msgf_m_Uns, msgf_W_Uns, y_obs, msgb_W_Yns, m_Yns);
        
        BD(N, M, L, K, m_X1, m_Uns, m_Yns, A, B, C, msgf_m_X1, msgf_W_X1, msgf_m_Uns, 
           msgf_W_Uns, y_obs, msgb_W_Yns, grad_X1, grad_Uns);

        max_index1 = cblas_idamax(M,grad_X1,1);
        max_index2 = cblas_idamax(N*L,grad_Uns,1);
        if(fabs(grad_X1[max_index1])>fabs(grad_Uns[max_index2])) gamma_max = fabs(grad_X1[max_index1]);
        else gamma_max = fabs(grad_Uns[max_index2]);
        
        return gamma_max;

       }

double PI(int N, int M, int L, int K, int maxit, double tol, double *m_X1, double *m_Uns, double *m_Yns, double *p_X1, 
          double *p_Uns, double *A, double *B, double *C, double *msgf_m_X1, double *msgf_W_X1, double *y_obs, 
          double *msgb_W_Yns, double *grad_X1, double *grad_Uns)
       {
                int it  = 0;
               double maxmum_eigen_value = -1;
               double maxmum_eigen_value_prev = 0.;
               double phi = 0.0;
               double X1_norm2 = 0.0;
               double Uns_norm2 = 0.0;
               double vec_norm_inverse = 0.0;




               double * msgf_m_Uns;
               double * msgf_W_Uns;
               double * grad_X1_aux;
               double * grad_Uns_aux;

               msgf_m_Uns= (double *) malloc(L* sizeof(double));
               memset(msgf_m_Uns,0,L*sizeof(double));

               msgf_W_Uns= (double *) malloc(L*L* sizeof(double));
               memset(msgf_W_Uns,0,L*L*sizeof(double));

               grad_X1_aux= (double *) malloc(M* sizeof(double));
               memset(grad_X1_aux,0,M*sizeof(double));

               grad_Uns_aux= (double *) malloc(N*L* sizeof(double));
               memset(grad_Uns_aux,0,N*L*sizeof(double));

               memset(m_X1,0,M*sizeof(double));
               memset(m_Uns,0,N*L*sizeof(double));



               phi = FE(N, M, L, K, 0., m_X1, m_Uns, p_X1, p_Uns, A, B, C, msgf_m_X1,
                        msgf_W_X1, msgf_m_Uns, msgf_W_Uns, y_obs, msgb_W_Yns, m_Yns);

               BD(N, M, L, K, m_X1, m_Uns, m_Yns, A, B, C, msgf_m_X1, msgf_W_X1, msgf_m_Uns,
                  msgf_W_Uns, y_obs, msgb_W_Yns, grad_X1_aux, grad_Uns_aux);



               m_X1[0] = 1.;

               while (it<maxit)
               {
                   if(fabs((maxmum_eigen_value-maxmum_eigen_value_prev)/maxmum_eigen_value_prev)<tol){
                       printf("\n Power method converges at %d-th iteration \n",it);
                       break;
                   }else{
                       maxmum_eigen_value_prev = maxmum_eigen_value;
                       X1_norm2 = cblas_ddot(M,m_X1,1,m_X1,1);
                       Uns_norm2 = cblas_ddot(N*L,m_Uns,1,m_Uns,1);
                       vec_norm_inverse = 1.0/sqrt(X1_norm2 + Uns_norm2);
                       cblas_dscal(M,vec_norm_inverse,m_X1,1);
                       cblas_dscal(N*L,vec_norm_inverse,m_Uns,1);

                       phi = FE(N, M, L, K, 0., m_X1, m_Uns, p_X1, p_Uns, A, B, C, msgf_m_X1,
                       msgf_W_X1, msgf_m_Uns, msgf_W_Uns, y_obs, msgb_W_Yns, m_Yns);

                       BD(N, M, L, K, m_X1, m_Uns, m_Yns, A, B, C, msgf_m_X1, msgf_W_X1, msgf_m_Uns,
                       msgf_W_Uns, y_obs, msgb_W_Yns, grad_X1, grad_Uns);

                       cblas_daxpy(M,-1,grad_X1_aux,1,grad_X1,1);
                       cblas_daxpy(N*L,-1,grad_Uns_aux,1,grad_Uns,1);

                       maxmum_eigen_value = cblas_ddot(M,m_X1,1,grad_X1,1);
                       maxmum_eigen_value = maxmum_eigen_value + cblas_ddot(N*L,m_Uns,1,grad_Uns,1);

                       cblas_dcopy(M,grad_X1,1,m_X1,1);
                       cblas_dcopy(N*L,grad_Uns,1,m_Uns,1);

                       it = it + 1;

                   }
               }



               free(msgf_m_Uns);
               free(msgf_W_Uns);
               free(grad_X1_aux);
               free(grad_Uns_aux);

               printf("\n maximum eigenvalue is %f \n",maxmum_eigen_value);

               return maxmum_eigen_value;
       }
       /*
       double PI(int N, int M, int L, int K, int maxit, double tol, double * m_X1, double * m_Uns, double * m_Yns, double *p_X1,
              double *p_Uns, double * A, double * B, double * C, double * msgf_m_X1, double * msgf_W_X1, double * y_obs,
              double * msgb_W_Yns, double * grad_X1, double * grad_Uns){

               int it  = 0;
               double maxmum_eigen_value = -1;
               double maxmum_eigen_value_prev = 0.;
               double phi = 0.0;
               double X1_norm2 = 0.0;
               double Uns_norm2 = 0.0;
               double vec_norm_inverse = 0.0;




               double * msgf_m_Uns;
               double * msgf_W_Uns;
               double * grad_X1_aux;
               double * grad_Uns_aux;

               msgf_m_Uns= (double *) malloc(L* sizeof(double));
               memset(msgf_m_Uns,0,L*sizeof(double));

               msgf_W_Uns= (double *) malloc(L*L* sizeof(double));
               memset(msgf_W_Uns,0,L*L*sizeof(double));

               grad_X1_aux= (double *) malloc(M* sizeof(double));
               memset(grad_X1_aux,0,M*sizeof(double));

               grad_Uns_aux= (double *) malloc(N*L* sizeof(double));
               memset(grad_Uns_aux,0,N*L*sizeof(double));

               memset(m_X1,0,M*sizeof(double));
               memset(m_Uns,0,N*L*sizeof(double));



               phi = FE(N, M, L, K, 0., m_X1, m_Uns, p_X1, p_Uns, A, B, C, msgf_m_X1,
                        msgf_W_X1, msgf_m_Uns, msgf_W_Uns, y_obs, msgb_W_Yns, m_Yns);

               BD(N, M, L, K, m_X1, m_Uns, m_Yns, A, B, C, msgf_m_X1, msgf_W_X1, msgf_m_Uns,
                  msgf_W_Uns, y_obs, msgb_W_Yns, grad_X1_aux, grad_Uns_aux);



               m_X1[0] = 1.;

               while (it<maxit)
               {
                   if(fabs((maxmum_eigen_value-maxmum_eigen_value_prev)/maxmum_eigen_value_prev)<tol){
                       printf("\n Power method converges at %d-th iteration \n",it);
                       break;
                   }else{
                       maxmum_eigen_value_prev = maxmum_eigen_value;
                       X1_norm2 = cblas_ddot(M,m_X1,1,m_X1,1);
                       Uns_norm2 = cblas_ddot(N*L,m_Uns,1,m_Uns,1);
                       vec_norm_inverse = 1.0/sqrt(X1_norm2 + Uns_norm2);
                       cblas_dscal(M,vec_norm_inverse,m_X1,1);
                       cblas_dscal(N*L,vec_norm_inverse,m_Uns,1);

                       phi = FE(N, M, L, K, 0., m_X1, m_Uns, p_X1, p_Uns, A, B, C, msgf_m_X1,
                       msgf_W_X1, msgf_m_Uns, msgf_W_Uns, y_obs, msgb_W_Yns, m_Yns);

                       BD(N, M, L, K, m_X1, m_Uns, m_Yns, A, B, C, msgf_m_X1, msgf_W_X1, msgf_m_Uns,
                       msgf_W_Uns, y_obs, msgb_W_Yns, grad_X1, grad_Uns);

                       cblas_daxpy(M,-1,grad_X1_aux,1,grad_X1,1);
                       cblas_daxpy(N*L,-1,grad_Uns_aux,1,grad_Uns,1);

                       maxmum_eigen_value = cblas_ddot(M,m_X1,1,grad_X1,1);
                       maxmum_eigen_value = maxmum_eigen_value + cblas_ddot(N*L,m_Uns,1,grad_Uns,1);

                       cblas_dcopy(M,grad_X1,1,m_X1,1);
                       cblas_dcopy(N*L,grad_Uns,1,m_Uns,1);

                       it = it + 1;

                   }
               }



               free(msgf_m_Uns);
               free(msgf_W_Uns);
               free(grad_X1_aux);
               free(grad_Uns_aux);

               printf("\n maximum eigenvalue is %f \n",maxmum_eigen_value);

               return maxmum_eigen_value;


       }
        */



