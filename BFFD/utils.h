#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

void print_1darray(double* arr, int n);
void print_2darray(double* arr, int nrow, int ncol);



double fsgn(double v);
double map_estimation(double msgf_m_U, double msgf_V_U, double msgb_m_U, double msgb_V_U);


/*
functions prepared for Gaussian distributions
*/

double gaussian_regularization(double u, double beta);

/*
functions prepared for Laplace distribution in NUP representation
*/

void laplace_unknown_parameters(double u, double beta, double* msgf_m_U, double* msgf_V_U);
double laplace_decision(double msgb_m_U, double msgb_V_U, double beta);
double laplace_regularization(double u, double beta);
void laplace_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double* u, double* msgf_m_U, double* msgf_V_U);
void laplace_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double* u, double* msgf_m_U, double* msgf_V_U);


/*
functions prepared for Huber loss in NUP representation
*/

void huber_unknown_parameters(double u, double beta, double r, double* msgf_m_U, double* msgf_V_U);
double huber_decision(double msgb_m_U, double msgb_V_U, double beta, double r);
double huber_regularization(double u, double beta, double r);
void huber_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double r, double* u, double* msgf_m_U, double* msgf_V_U);
void huber_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double r, double* u, double* msgf_m_U, double* msgf_V_U);


/*
functions prepared for Hinge loss in NUP representation
*/

void hinge_unknown_parameters(double u, double beta, double a, double* msgf_m_U, double* msgf_V_U);
double hinge_decision(double msgb_m_U, double msgb_V_U, double beta, double a);
double hinge_regularization(double u, double beta, double a);
void hinge_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double* u, double* msgf_m_U, double* msgf_V_U);
void hinge_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double* u, double* msgf_m_U, double* msgf_V_U);
double hinge_adaptive_parameter(double u,double msgb_m_U, double msgb_V_U, double a);



/*
functions prepared for Vapnik loss in NUP representation
*/

void vapnik_unknown_parameters(double u, double beta, double a, double b, double* msgf_m_U, double* msgf_V_U);
double vapnik_decision(double msgb_m_U, double msgb_V_U, double beta, double a, double b);
double vapnik_regularization(double u, double beta, double a, double b);
void vapnik_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double b,double* u, double* msgf_m_U, double* msgf_V_U);
void vapnik_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double b,double* u, double* msgf_m_U, double* msgf_V_U);
double vapnik_adaptive_parameter(double u,double msgb_m_U, double msgb_V_U, double a, double b);


/*
functions prepared for PlainNUV distribution whose deciding rules are derived from Type-II MAP estimation.
*/

void plainNUV_unknown_parameters(double u, double* msgf_m_U, double* msgf_V_U);
double plainNUV_decision(double msgb_m_U, double msgb_V_U);
double plainNUV_regularization(double u);
void plainNUV_alternating_updating(double msgb_m_U, double msgb_V_U,double* u, double* msgf_m_U, double* msgf_V_U);
void plainNUV_deciding_updating(double msgb_m_U, double msgb_V_U,double* u, double* msgf_m_U, double* msgf_V_U);







#endif