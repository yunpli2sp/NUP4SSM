#ifndef BFFD_SMOOTHING_H_INCLUDED
#define BFFD_SMOOTHING_H_INCLUDED

#include "utils.h"

int solve_equations(int n, double * A, double * b);
int initial_state(int M,double * msgf_Xi_X1, double * msgf_W_X1, double * msgb_Xi_X1, double * msgb_W_X1, double * sol);
void NUP_unknown_parameters(int choice, double u, double beta, double* NUP_parameters, double *msgf_m_U, double *msgf_V_U);
double NUP_decisions(int choice, double msgb_m_U, double msgb_V_U, double beta, double* NUP_parameters);
double NUP_regularizations(int choice, double m_U, double beta, double* NUP_parameters);

int bffd_input_estimation(int choice, double * NUP_parameters, double beta, int N, int M, int L, int K, int max_valid_index,
double *A, double *B, double *C, double *y_obs, double *msgf_Xi_X1, double *msgf_W_X1,double *msgb_Xi_Xf, double *msgb_W_Xf,
double *msgb_W_Yn, int maxit, double obj_tol, double eps, double *m_X1, double *m_Us, double *m_Ys, double *objs);


int bffd_constrained_control(int choice, double * NUP_parameters, int N, int M, int L, double * A, double * B, double *Q, double *Qf, 
double * x1, double *xf, int maxit, double obj_tol, double eps,double *m_Xs,double *m_Us,double *objs);

#endif