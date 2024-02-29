/* utils.c
 *
 * Copyright (C) 2024 Yun-Peng Li, Hans-Andrea Loeliger.
 *
 * Date: 2024-2-13
 * Author: Yunpeng Li (yunpli@isi.ee.ethz.ch)
 * Brief: Fundamental functions designed for NUP (normal with unknown parameters) priors:
 *               1) the updating rules of unknown parameters;
 *               2) the deciding rules of posterior estimation;
 *               3) the adjusting rules of scale parameters used in constraint satisfication.
 * 
 * Cite: Y.-P Li, H.-A. Loeliger, “Backward Filtering Forward Deciding in Linear Non-Gaussian 
 *        State Space Models,” (to be appeared) in 2024 International Conference on Artificial 
 *        Intelligence and Statistics (AISTATS), May 2024.
 */

#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#include "utils.h"

void print_1darray(double* arr, int n){
  int i = 0;
  printf("\n");
  for(i=0;i<n;i++) printf(" %.6f\t", arr[i]);
  printf("\n");
}



void print_2darray(double* arr, int nrow, int ncol){


 int i = 0;
 int j = 0;

 for(i=0;i<nrow;i++){
  printf("\n");
  for(j=0;j<ncol;j++) printf(" %.6f\t", arr[i*ncol+j]);
 }
 printf("\n");
}

double fsgn(double v){
    if(v<0.){
        return -1.;
    }else if(v>0.){
        return 1.;
    }else{
        return 0;
    }
}

double map_estimation(double msgf_m_U, double msgf_V_U, double msgb_m_U, double msgb_V_U){
   return (msgb_V_U*msgf_m_U + msgf_V_U*msgb_m_U)/(msgb_V_U + msgf_V_U);
}

double gaussian_regularization(double u, double beta)
{
    return beta*u*u;
}

void laplace_unknown_parameters(double u, double beta, double *msgf_m_U, double *msgf_V_U)
{   *msgf_m_U = 0.0;
    *msgf_V_U = fabs(u)/beta;
    return;
}

double laplace_decision(double msgb_m_U, double msgb_V_U, double beta)
{
    return fsgn(msgb_m_U)*fmax(fabs(msgb_m_U)-beta*msgb_V_U,0.);
    
}

double laplace_regularization(double u, double beta)
{
    return beta*fabs(u);
}

void laplace_alternating_updating(double msgb_m_U, double msgb_V_U, double beta,double* u, double* msgf_m_U, double* msgf_V_U)
{   *u = map_estimation(*msgf_m_U,*msgf_V_U,msgb_m_U,msgb_V_U);
    laplace_unknown_parameters(*u,beta,msgf_m_U, msgf_V_U);
    return;
}

void laplace_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double* u, double* msgf_m_U, double* msgf_V_U)
{
    *u = laplace_decision(msgb_m_U,msgb_V_U,beta);
    laplace_unknown_parameters(*u,beta,msgf_m_U, msgf_V_U);
    return;
}

void huber_unknown_parameters(double u, double beta, double r, double *msgf_m_U, double *msgf_V_U)
{   *msgf_m_U = 0.0;
    if(fabs(u)<=(beta*r*r)){
        *msgf_V_U = r*r;
    }
    else{
        *msgf_V_U = fabs(u)/beta;
    }

    return;
}

double huber_decision(double msgb_m_U, double msgb_V_U, double beta, double r)
{
    if(msgb_m_U<(-beta*(msgb_V_U+r*r))){
        return msgb_m_U + beta*msgb_V_U;
    }else if(msgb_m_U>(beta*(msgb_V_U+r*r))){
        return msgb_m_U - beta*msgb_V_U;
    }else{
        return (r*r*msgb_m_U)/(msgb_V_U+r*r);
    }

}

double huber_regularization(double u, double beta, double r)
{
    if(fabs(u)<=(beta*r*r)){
        return ((u*u)/(2*r*r)) + ((beta*beta*r*r)/2.0);
    }
    else{
        return  beta*fabs(u);
    }

}

void huber_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double r, double *u, double *msgf_m_U, double *msgf_V_U)
{

    *u = map_estimation(*msgf_m_U,*msgf_V_U,msgb_m_U,msgb_V_U);
    huber_unknown_parameters(*u,beta,r, msgf_m_U, msgf_V_U);
    return;
}

void huber_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double r, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = huber_decision(msgb_m_U, msgb_V_U, beta, r);
    huber_unknown_parameters(*u,beta,r, msgf_m_U, msgf_V_U);
    return;
}

void hinge_unknown_parameters(double u, double beta, double a, double *msgf_m_U, double *msgf_V_U)
{
    *msgf_m_U = a+ fabs(u-a);
    *msgf_V_U = 2*fabs(u-a)/beta;
    return;
}

double hinge_decision(double msgb_m_U, double msgb_V_U, double beta, double a)
{
    if(msgb_m_U>a){
        return msgb_m_U;
    }else if(msgb_m_U<(-beta*msgb_V_U+a)){
        return msgb_m_U+beta*msgb_V_U;
    }else{
        return a;
    }

}

double hinge_regularization(double u, double beta, double a)
{
    return beta*fabs(fmin(u-a,0));
}

void hinge_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = map_estimation(*msgf_m_U,*msgf_V_U,msgb_m_U,msgb_V_U);
    hinge_unknown_parameters(*u, beta, a, msgf_m_U, msgf_V_U);
    return;

}

void hinge_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = hinge_decision(msgb_m_U, msgb_V_U, beta,a);
    hinge_unknown_parameters(*u, beta, a, msgf_m_U, msgf_V_U);
    return;
}

double hinge_adaptive_parameter(double u, double msgb_m_U, double msgb_V_U, double a)
{
    if(u<a){
        return (a-msgb_m_U)/msgb_V_U;
    }else{
        return -1.;
    }

}

void vapnik_unknown_parameters(double u, double beta, double a, double b, double *msgf_m_U, double *msgf_V_U)
{
    *msgf_m_U = (a*fabs(u-b)+b*fabs(u-a))/(fabs(u-a)+fabs(u-b));
    *msgf_V_U = (fabs(u-a)*fabs(u-b))/(beta*(fabs(u-a)+fabs(u-b)));
}

double vapnik_decision(double msgb_m_U, double msgb_V_U, double beta, double a, double b)
{
    if(msgb_m_U<(-2*beta*msgb_V_U+a)){
        return msgb_m_U+2*beta*msgb_V_U;
    }else if(msgb_m_U>= ((-2*beta*msgb_V_U+a)) && msgb_m_U<=a){
        return a;
    }else if(msgb_m_U>a && msgb_m_U<b){
        return msgb_m_U;
    }else if(msgb_m_U>=b && msgb_m_U<=(2*beta*msgb_V_U+b)){
        return b;
    }else{
        return msgb_m_U-2*beta*msgb_V_U;
    }

}

double vapnik_regularization(double u, double beta, double a, double b)
{
    return beta*(fabs(u-a)+fabs(u-b));
}

void vapnik_alternating_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double b, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = map_estimation(*msgf_m_U,*msgf_V_U,msgb_m_U,msgb_V_U);
    vapnik_unknown_parameters(*u, beta, a, b, msgf_m_U,msgf_V_U);
    return;
}

void vapnik_deciding_updating(double msgb_m_U, double msgb_V_U, double beta, double a, double b, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = vapnik_decision(msgb_m_U,msgb_V_U,beta, a,b);
    vapnik_unknown_parameters(*u, beta, a, b, msgf_m_U,msgf_V_U);
}

double vapnik_adaptive_parameter(double u, double msgb_m_U, double msgb_V_U, double a, double b)
{
    if(u<a){
        return (a-msgb_m_U)/(2*msgb_V_U);
    }else if(u>b){
        return (msgb_m_U-b)/(2*msgb_V_U);
    }else{
        return -1.;
    }

}

void plainNUV_unknown_parameters(double u, double *msgf_m_U, double *msgf_V_U)
{
    *msgf_m_U = 0.0;
    *msgf_V_U = u*u;
    return;
}

double plainNUV_decision(double msgb_m_U, double msgb_V_U)
{
    if((msgb_m_U*msgb_m_U)>msgb_V_U){
        return msgb_m_U - (msgb_V_U/msgb_m_U);
    }else{
        return 0.0;
    }

}

double plainNUV_regularization(double u)
{
    return log(fabs(u));
}

void plainNUV_alternating_updating(double msgb_m_U, double msgb_V_U, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = map_estimation(*msgf_m_U,*msgf_V_U,msgb_m_U,msgb_V_U);
    plainNUV_unknown_parameters(*u, msgf_m_U,msgf_V_U);
    return;
}

void plainNUV_deciding_updating(double msgb_m_U, double msgb_V_U, double *u, double *msgf_m_U, double *msgf_V_U)
{
    *u = plainNUV_decision(msgb_m_U, msgb_V_U);
    plainNUV_unknown_parameters(*u, msgf_m_U,msgf_V_U);
    return;

}

int main(){
    double msgf_m_U = -1.;
    double msgf_V_U = 1.;
    double msgb_m_U = -2.;
    double msgb_V_U = 1.;
    printf("\n Hello World %f \n ", map_estimation(msgf_m_U, msgf_V_U, msgb_m_U, msgb_V_U));
}
