#include<math.h>
#include<stdio.h>
#include <stdlib.h>
#include "nup_rules.h"

double fsgn(double v){
    if(v<0.){
        return -1.;
    }else if(v>0.){
        return 1.;
    }else{
        return 0;
    }
}

double nup_regularizer(int choice,double z, double beta, double a, double b){
    if(choice==1){
        return beta*fabs(z-a);
    }else if(choice==2){
        return beta*fmax(a-z,0.);
    }else if(choice==3){
        return beta*fmax(z-b,0.);
    }else if(choice==4){
        return beta*(fabs(z-a)+fabs(z-b)-fabs(a-b));
    }else{
        return 0.5*beta*(z-a)*(z-a);
    }
}

void nup_primal_unknown_parameters(int choice, double z, double beta, double a, double b, double* intrinsic_mean, double* intrinsic_var){
    if(choice==1){
        *intrinsic_mean = a;
        *intrinsic_var = fabs(z-a)/beta;
    }else if(choice==2){
        *intrinsic_mean = a + fabs(a-z);
        *intrinsic_var = 2*fabs(z-a)/beta;
    }else if(choice==3){
        *intrinsic_mean = b - fabs(z-b);
        *intrinsic_var = 2*fabs(z-b)/beta;
    }else if(choice==4){
        *intrinsic_mean= (a*fabs(z-b)+b*fabs(z-a))/(fabs(z-a)+fabs(z-b));
        *intrinsic_var = (fabs(z-a)*fabs(z-b))/(beta*(fabs(z-a)+fabs(z-b)));
    }else{
        *intrinsic_mean= a;
        *intrinsic_var = 1/beta;
    }
}

void nup_dual_unknown_parameters(int choice, int is_constrained, double gamma, double dual_z, double beta, double a, double b, double *intrinsic_precision_mean, double *intrinsic_precision_var)
{
    if(gamma>1e10){
        *intrinsic_precision_mean = -dual_z;
        *intrinsic_precision_var = 0.;
    }else{
        if(choice==1){
             *intrinsic_precision_mean = (a*fabs((dual_z+beta)*(dual_z-beta))+beta*gamma*(fabs(dual_z-beta)-fabs(dual_z+beta)))/(gamma*(fabs(dual_z+beta)+fabs(dual_z-beta)));
             *intrinsic_precision_var = fabs((dual_z+beta)*(dual_z-beta))/(gamma*(fabs(dual_z+beta)+fabs(dual_z-beta)));
        }else if(choice==2){
            if(is_constrained>0){
                *intrinsic_precision_mean = ((a+gamma)*fabs(dual_z))/gamma;
                *intrinsic_precision_var = (fabs(dual_z))/gamma;
            }else{
                *intrinsic_precision_mean = (a*fabs((dual_z+beta)*dual_z)+beta*gamma*fabs(dual_z))/(gamma*(fabs(dual_z+beta)+fabs(dual_z)));
                *intrinsic_precision_var = (fabs((dual_z+beta)*dual_z))/(gamma*(fabs(dual_z+beta)+fabs(dual_z)));
            }
        }else if(choice==3){
            if(is_constrained>0){
                *intrinsic_precision_mean = ((b-gamma)*fabs(dual_z))/gamma;
                *intrinsic_precision_var = (fabs(dual_z))/gamma;
            }else{
                *intrinsic_precision_mean = (b*fabs(dual_z*(dual_z-beta))-beta*gamma*fabs(dual_z))/(gamma*(fabs(dual_z)+fabs(dual_z-beta)));
                *intrinsic_precision_var = (fabs(dual_z*(dual_z-beta)))/(gamma*(fabs(dual_z)+fabs(dual_z-beta)));
            }
        }else if(choice==4){
            if(dual_z<=0){
                if(is_constrained>0){
                    *intrinsic_precision_mean = ((a+gamma)*fabs(dual_z))/gamma;
                    *intrinsic_precision_var = (fabs(dual_z))/gamma;
                }else{
                    *intrinsic_precision_mean = (a*fabs((dual_z+2*beta)*dual_z)+2*beta*gamma*fabs(dual_z))/(gamma*(fabs(dual_z+2*beta)+fabs(dual_z)));
                    *intrinsic_precision_var = fabs((dual_z+2*beta)*dual_z)/(gamma*(fabs(dual_z+2*beta)+fabs(dual_z)));
                }
            }else{
                if(is_constrained>0){
                    *intrinsic_precision_mean = ((b-gamma)*fabs(dual_z))/gamma;
                    *intrinsic_precision_var = (fabs(dual_z))/gamma;
                }else{
                    *intrinsic_precision_mean = (b*fabs((dual_z-2*beta)*dual_z)-2*beta*gamma*fabs(dual_z))/(gamma*(fabs(dual_z-2*beta)+fabs(dual_z)));
                    *intrinsic_precision_var = fabs((dual_z-2*beta)*dual_z)/(gamma*(fabs(dual_z-2*beta)+fabs(dual_z)));
                }
            }
        }else{
             *intrinsic_precision_mean= beta*a;
             *intrinsic_precision_var  = beta;
        }
    }
}

double nup_primal_deciding(int choice, double extrinsic_mean, double extrinsic_var, double beta, double a, double b){
    if(choice==1){
        if(extrinsic_mean<(-beta*extrinsic_var+a)) return extrinsic_mean + beta*extrinsic_var;
        else if(extrinsic_mean>(beta*extrinsic_var+a)) return extrinsic_mean - beta*extrinsic_var;
        else return a;
    }else if(choice==2){
        if(extrinsic_mean<(-beta*extrinsic_var+a)) return extrinsic_mean + beta*extrinsic_var;
        else if(extrinsic_mean>a) return extrinsic_mean;
        else return a;
    }else if(choice==3){
        if(extrinsic_mean>(beta*extrinsic_var+b)) return extrinsic_mean - beta*extrinsic_var;
        else if(extrinsic_mean<b) return extrinsic_mean;
        else return b;
    }else if(choice==4){
        if(extrinsic_mean<(-2*beta*extrinsic_var+a)) return extrinsic_mean+2*beta*extrinsic_var;
        else if(extrinsic_mean>= ((-2*beta*extrinsic_var+a)) && extrinsic_mean<=a) return a;
        else if(extrinsic_mean>a && extrinsic_mean<b) return extrinsic_mean;
        else if(extrinsic_mean>=b && extrinsic_mean<=(2*beta*extrinsic_var+b)) return b;
        else return extrinsic_mean-2*beta*extrinsic_var;
    }else{
        return (extrinsic_mean+beta*a*extrinsic_var)/(1+beta*extrinsic_var);
    }
}

double nup_dual_deciding(int choice, int is_constrained, double extrinsic_precision_mean, double extrinsic_precision, double beta, double a, double b){
    if(choice==1){
        if(extrinsic_precision_mean<(a*extrinsic_precision-beta)) return -beta;
        else if(extrinsic_precision_mean>(a*extrinsic_precision+beta)) return beta;
        else return extrinsic_precision_mean - a*extrinsic_precision;
    }else if(choice==2){
        if(is_constrained>0){
            return fmin(extrinsic_precision_mean - a*extrinsic_precision,0);
        }else{
            if(extrinsic_precision_mean<(a*extrinsic_precision-beta)) return -beta;
            else if(extrinsic_precision_mean>(a*extrinsic_precision)) return 0.;
            else return extrinsic_precision_mean - a*extrinsic_precision;
        }
    }else if(choice==3){
        if(is_constrained>0){
            return fmax(extrinsic_precision_mean - b*extrinsic_precision,0);
        }else{
            if(extrinsic_precision_mean>(b*extrinsic_precision+beta)) return beta;
            else if(extrinsic_precision_mean<(b*extrinsic_precision)) return 0.;
            else return extrinsic_precision_mean - b*extrinsic_precision; 
        }
    }else if(choice==4){
        if(is_constrained>0){

            if(extrinsic_precision_mean<=a*extrinsic_precision) return extrinsic_precision_mean - a*extrinsic_precision;
            else if(extrinsic_precision_mean>(a*extrinsic_precision) && extrinsic_precision_mean<(b*extrinsic_precision)) return 0.;
            else return extrinsic_precision_mean - b*extrinsic_precision;       
        }
        else{
            if(extrinsic_precision_mean<(a*extrinsic_precision-2*beta)) return -2*beta;
            else if(extrinsic_precision_mean>=(a*extrinsic_precision-2*beta) && extrinsic_precision_mean<=a*extrinsic_precision) return extrinsic_precision_mean - a*extrinsic_precision;
            else if(extrinsic_precision_mean>(a*extrinsic_precision) && extrinsic_precision_mean<(b*extrinsic_precision)) return 0.;
            else if(extrinsic_precision_mean>=(b*extrinsic_precision) && extrinsic_precision_mean<=(b*extrinsic_precision+2*beta)) return extrinsic_precision_mean - b*extrinsic_precision;
            else return 2*beta;
        }
    }else{
        return (beta*(extrinsic_precision_mean - a*extrinsic_precision))/(beta+extrinsic_precision);
    }
}


double nup_dual_adaptive_parameter(int choice, int is_constrained, double beta,double extrinsic_precision_mean, double extrinsic_precision, double a, double b)
{
    
    if(choice==1) return 0.5*fmax((a-((extrinsic_precision_mean+beta)/extrinsic_precision)),(((extrinsic_precision_mean-beta)/extrinsic_precision)-a));
    else if(choice==2){
        if(is_constrained>0){
            return 0.5*(((extrinsic_precision_mean)/extrinsic_precision)-a);
        }else{
            return 0.5*fmax((a-((extrinsic_precision_mean+beta)/extrinsic_precision)),(((extrinsic_precision_mean)/extrinsic_precision)-a));
        }
    }
    else if(choice==3){
        if(is_constrained>0){
            return 0.5*(b-((extrinsic_precision_mean)/extrinsic_precision));
        }else{
            return 0.5*fmax((b-((extrinsic_precision_mean)/extrinsic_precision)),(((extrinsic_precision_mean-beta)/extrinsic_precision)-b));
        }
    } 
    else if(choice==4){
        if(is_constrained>0){
            //return 0.25*fabs(b-a);
            if((extrinsic_precision_mean>a*extrinsic_precision) && (extrinsic_precision_mean<b*extrinsic_precision)){
                //if(extrinsic_precision_mean>0.5*(a+b)*extrinsic_precision) return 0.5*(((extrinsic_precision_mean)/extrinsic_precision)-a);
                //else return 0.5*(b-((extrinsic_precision_mean)/extrinsic_precision));
                return 0.5*fmin((((extrinsic_precision_mean)/extrinsic_precision)-a),(b-((extrinsic_precision_mean)/extrinsic_precision)));
            }else return 0.25*fabs(b-a);
            // return 0.5*fmin((((extrinsic_precision_mean)/extrinsic_precision)-a),(b-((extrinsic_precision_mean)/extrinsic_precision)));
            //return 0.5*fabs(b-a);
            //return 0.5*fmin((((extrinsic_precision_mean)/extrinsic_precision)-a),(b-((extrinsic_precision_mean)/extrinsic_precision)));
            //return 0.5*(((extrinsic_precision_mean)/extrinsic_precision)-a);
            //return 0.5*(b-((extrinsic_precision_mean)/extrinsic_precision));
        }else{
            return 0.5*fmax((a-((extrinsic_precision_mean+2*beta)/extrinsic_precision)),(((extrinsic_precision_mean-2*beta)/extrinsic_precision)-b));
        }
    } 
    else return 0.;

}


int nup_state(double z, double a, double b, double eps)
{
    if(fabs(z-a)<eps) return 1;
    else if(fabs(z-b)<eps) return 2;
    else return 0;
}