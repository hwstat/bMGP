#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Last revised: 3/5/2023

// [[Rcpp::export]]
double LogLikeC(mat DQP, vec all_quantiles, vec y, vec mu_y, vec sigma_y, vec group_ind) 
{
  int n = y.n_elem;
  vec y_unif(n);

  double nrows = all_quantiles.n_elem;
  double log_like = 0;
  
  for(int i = 0; i < n; ++i) {
    y_unif(i) = R::pnorm(y(i), mu_y(group_ind(i)), sigma_y(group_ind(i)), true, false);

    vec DQP_col(nrows);
    double t = 0;
    for (int j = 0; j < nrows; j++){
    
    // DQP_col calculation //
      DQP_col(j) = DQP(j, group_ind(i));
      
      // t calculation //
      if(y_unif(i) > DQP_col(j)){
        t++;
      }
    } 
    
    if((t == 0) | (t == nrows)){ 
      // if t value is odd due to extreme a y_unif value //
      log_like = sqrt(-1); // NaN
      break;
    }

    double num = all_quantiles(t) - all_quantiles(t-1);
    double denom = DQP_col(t) - DQP_col(t-1);
    double log_phi_y = R::dnorm(y(i), mu_y(group_ind(i)), sigma_y(group_ind(i)), true); // ** Fixed as of 3/5/2023
    double log_f = log(num) - log(denom) + log_phi_y;

    log_like += log_f;
  }
  
  return log_like;
}

