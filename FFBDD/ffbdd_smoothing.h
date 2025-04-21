#ifndef BFFDD_SMOOTHING_H_INCLUDED
#define BFFDD_SMOOTHING_H_INCLUDED

#include "dual_utils.h"

double evaluate_quadratic_objective(int M, double * m_X, double * mean_X, double * W_X);

int iterated_ffbdd_nonGaussian_output(int choice, int is_constrained, int N, int M, int L, int K, double *lower_bounds, double * upper_bounds, double beta, 
double *A, double *B, double *C, double *msgf_m_X1, double *msgf_V_X1, double * msgf_W_X1,double *dual_xi_Xf, double *dual_W_Xf, double * msgf_m_Uns,
double * msgf_V_Uns, double * msgf_W_Uns, int maxit, double obj_tol, double init_gamma, double *m_X1, double *m_Us, double *m_Ys, double *objs);

#endif