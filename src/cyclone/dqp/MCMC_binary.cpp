// Dependent quantile pyramid sampler of Hyoin An and Steven MacEachern.
// This file and the headers next to it are copied unchanged from
// run_scripts/7.2_cyclone_dqp/ver0/ of the qmp repository
// (https://github.com/edfong/qmp), which itself carries Hyoin An's DQP code
// from https://github.com/hyoin-an/DQP. Compiled at run time by
// src/cyclone/time_dqp.R through Rcpp::sourceCpp.
//
// MIT License, Copyright (c) 2023 Hyoin An. The full notice is in
// src/cyclone/dqp/LICENSE.

#include <RcppArmadillo.h>
#include "utilities.h"
#include "DQPpriorSampling_functions.h" // not used in mcmc, but used to generate an initial value //
#include "DQPdensity_functions.h"
// #include "Likelihood_functions.h"
#include "Likelihood_functions_insensitive.h"
#include "UpdateQ_binary.h"
#include "UpdateBeta_binary.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
mat unique_rows(const mat& x) {
  
  // Initialize matrix for unique rows
  mat unique_x = x.rows(0, 0);
  
  for (uword i = 1; i < x.n_rows; ++i) {
    bool is_unique = true;
    for (uword j = 0; j < unique_x.n_rows; ++j) {
      if (all(x.row(i) == unique_x.row(j))) {
        is_unique = false;
        break;
      }
    }
    if (is_unique) {
      unique_x = join_vert(unique_x, x.rows(i, i));
    }
  }
  
  return unique_x;
}

// [[Rcpp::export]]
vec unique_rows_identifier(const mat& x) {
  
  // Create a mapping of unique rows to their identifiers
  std::unordered_map<std::string, uword> unique_row_map;
  uword next_id = 1;
  
  // Create a vector of identifiers for each row
  vec id_vec(x.n_rows);
  for (uword i = 0; i < x.n_rows; ++i) {
    std::string row_str;
    for (uword j = 0; j < x.n_cols; ++j) {
      row_str += std::to_string(x(i,j));
    }
    auto iter = unique_row_map.find(row_str);
    if (iter == unique_row_map.end()) {
      // This is a new unique row
      unique_row_map[row_str] = next_id++;
    }
    id_vec(i) = unique_row_map[row_str];
  }
  
  return id_vec-1; // to make the first index = 0, instead of 1
}

// [[Rcpp::export]]
List MCMC_param2(int niter, vec y, mat x, vec sigma_y, double delta, mat Sigma_beta, 
                 NumericMatrix DQP_Q_initial_rcpp, mat DQP_Z_initial, vec beta_initial, List quantile_levels, 
                 vec mu, mat Sigma, vec mu_0, mat Sigma_0, double alpha_scale=1, int n_chunk_Q=10, int n_chunk_beta=1){
    
    // Getting 'all_quantiles' vector
    vec all_quantiles = as_numeric(rownames(DQP_Q_initial_rcpp));
    mat DQP_Q_initial = as<mat>(DQP_Q_initial_rcpp);

    // Handling design matrix //
    mat X = join_horiz(ones(y.n_elem), x); // Design matrix with replicates
    mat X_prime = unique_rows(X); // Design matrix without replicates
    vec group_ind = unique_rows_identifier(X); 

    // For efficient computation //
    mat Sigma_inv = inv(Sigma);
    double log_Sigma_det = log(det(Sigma)); // this det() function needs to be checked
    mat Sigma_0_inv = inv(Sigma_0);
    double log_Sigma_0_det = log(det(Sigma_0));

    // initial values //
    mat DQP_Q_post = DQP_Q_initial;
    mat DQP_Z_post = DQP_Z_initial;
    vec beta_post = beta_initial;

    List Q_posteriors(niter);
    List beta_posteriors(niter);

    // iteration //
    for(int i=0; i<niter; i++){

        vec mu_y = X_prime * beta_post; // # need this for each group, not individually

        List DQP_post = UpdateQ(DQP_Q_post, DQP_Z_post, beta_post, y, mu_y, sigma_y, group_ind, quantile_levels, all_quantiles,
                                mu, Sigma, Sigma_inv, log_Sigma_det, delta, alpha_scale, n_chunk_Q);
        
        DQP_Q_post = as<mat>(DQP_post["Q"]);
        DQP_Z_post = as<mat>(DQP_post["Z"]);

        beta_post = UpdateBeta(DQP_Q_post, beta_post, X_prime, y, sigma_y, group_ind, quantile_levels, all_quantiles,
                               mu, Sigma, Sigma_inv, log_Sigma_det, alpha_scale, Sigma_beta, mu_0, Sigma_0_inv, log_Sigma_0_det, n_chunk_beta);
        
        Q_posteriors[i] = DQP_Q_post; 
        beta_posteriors[i] = beta_post;
    }

    List posterior_all = List::create(Named("Q") = Q_posteriors, Named("beta") = beta_posteriors);

    return posterior_all;
}
