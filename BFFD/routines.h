#ifndef ROUTINES_H_INCLUDED
#define ROUTINES_H_INCLUDED


double evaluate_least_squares(int M, double * m_X, double * mean_X, double * W_X);

double FE(int N, int M, int L, int K, double alpha, double *m_X1, double *m_Uns, double *p_X1, double *p_Uns,double *A, 
         double *B, double *C, double *msgf_m_X1, double *msgf_W_X1, double *msgf_m_Uns, double *msgf_W_Uns, double *y_obs, 
         double *msgb_W_Yns, double *m_Yns);

void BD(int N, int M, int L, int K, double * m_X1, double * m_Uns, double * m_Yns, double * A, double * B,
        double * C, double * msgf_m_X1, double * msgf_W_X1, double * msgf_m_Uns, double * msgf_W_Uns, 
        double * y_obs, double * msgb_W_Yns, double * grad_X1, double * grad_Uns);


double QI(int N, int M, int L, int K, double phi0, double phi0_der_1st, double alpha1, double *m_X1, double *m_Uns, 
          double *p_X1, double *p_Uns, double *m_Yns, double *A, double *B, double *C, double *msgf_m_X1, double *msgf_W_X1, 
          double *msgf_m_Uns, double *msgf_W_Uns, double *y_obs, double *msgb_W_Yns);

double maximum_scale_parameter(int N, int M, int L, int K, double * m_X1, double * m_Uns, double * m_Yns, double *p_X1, 
       double *p_Uns, double * A, double * B, double * C, double * msgf_m_X1, double * msgf_W_X1, double * y_obs, 
       double * msgb_W_Yns, double * grad_X1, double * grad_Uns);

double PI(int N, int M, int L, int K, int maxit, double tol, double * m_X1, double * m_Uns, double * m_Yns, double *p_X1, 
       double *p_Uns, double * A, double * B, double * C, double * msgf_m_X1, double * msgf_W_X1, double * y_obs, 
       double * msgb_W_Yns, double * grad_X1, double * grad_Uns);

       



#endif