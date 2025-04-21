#ifndef DUAL_UTILS_H_INCLUDED
#define DUAL_UTILS_H_INCLUDED

double fsgn(double v);

double nup_loss_function(int choice,double z, double beta, double a, double b);


void nup_dual_unknown_parameters(int choice, int is_constrained,double gamma, double dual_z, double beta, double a, double b, double *intrinsic_precision_mean, double *intrinsic_precision_var);

double nup_dual_deciding(int choice, int is_constrained, double extrinsic_precision_mean, double extrinsic_precision, double beta, double a, double b);

double nup_dual_adaptive_parameter(int choice, int is_constrained, double beta,double extrinsic_precision_mean, double extrinsic_precision, double a, double b);




#endif