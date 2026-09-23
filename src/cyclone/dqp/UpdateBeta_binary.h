#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Last revised: 3/2/2023 

// [[Rcpp::export]]
vec proposeBeta_chunk(uvec ind, vec beta_c, mat Sigma_beta)
{
    // Subsetting Sigma_beta //
    vec rowind = zeros<vec>(Sigma_beta.n_rows);
    vec colind = zeros<vec>(Sigma_beta.n_cols);
    rowind.elem(ind).fill(1);
    colind.elem(ind).fill(1);
    mat Sigma11 = Sigma_beta.submat(ind, ind); 
    mat Sigma12 = Sigma_beta.submat(arma::find(rowind==1), arma::find(colind==0));
    mat Sigma21 = Sigma12.t();
    mat Sigma22 = Sigma_beta.submat(arma::find(rowind==0), arma::find(colind==0));
    mat Sigma22_inv = inv(Sigma22);

    // conditional parameters //
    vec cond_mean = beta_c.elem(ind);
    mat cond_var = Sigma11 - Sigma12 * Sigma22_inv * Sigma21;

    // sample a chunk //
    vec beta_p_ind = sampleZ(cond_mean, cond_var);

    return beta_p_ind;
}

// [[Rcpp::export]]
double LogBetaPrior(vec beta, vec mu_0, mat Sigma_0_inv, double log_Sigma_0_det)
{
    int n = beta.size();
    vec diff = beta - mu_0;
    double quad_form = dot(diff, Sigma_0_inv * diff);
    double log_density = -0.5 * n * log(2 * M_PI) - 0.5 * log_Sigma_0_det - 0.5 * quad_form;
    return log_density;
}

// [[Rcpp::export]]
vec UpdateBeta(mat DQP, vec beta_c, mat X, vec y, vec sigma_y, vec group_ind, List quantile_levels, vec all_quantiles,
               vec mu, mat Sigma, mat Sigma_inv, double log_Sigma_det,  double alpha_scale, mat Sigma_beta, 
               vec mu_0, mat Sigma_0_inv, double log_Sigma_0_det, int n_chunk=1)
{
    // X = design matrix without replicates including intercept column

    uvec ind_chunks = chunk_indices(beta_c.n_elem, n_chunk);
    // update chunks (lower-dimensional update) //
    for (uword i = 0; i < ind_chunks.n_elem-1; i++){
        vec beta_p = beta_c;

        // (1) Propose a new candidate  
        uvec ind = regspace<uvec>(ind_chunks(i), ind_chunks(i+1)-1);
        vec beta_p_ind = proposeBeta_chunk(ind, beta_c, Sigma_beta);

        beta_p(ind) = beta_p_ind;

        // (2) Calculate (log) acceptance probability: a_star
        vec mu_y_p = X * beta_p;
        vec mu_y_c = X * beta_c;

        mat DQP_u_p(DQP.n_rows, DQP.n_cols);
        mat DQP_u_c(DQP.n_rows, DQP.n_cols);
        for(int r=0; r < DQP.n_rows; r++){
            DQP_u_p.row(r) = pnorm_cpp(DQP.row(r).t(), mu_y_p, sigma_y).t();
            DQP_u_c.row(r) = pnorm_cpp(DQP.row(r).t(), mu_y_c, sigma_y).t();
        }

        double log_like_p = LogLikeC(DQP_u_p, all_quantiles, y, mu_y_p, sigma_y, group_ind);
        double log_like_c = LogLikeC(DQP_u_c, all_quantiles, y, mu_y_c, sigma_y, group_ind);

        // fix from here
        // we need to include these lines b/c prior of Q (uniform/y-scale) does depend on beta 
        double log_Q_density_given_beta_p = DQPlogPrior_binary(DQP_u_p, quantile_levels, all_quantiles, mu_y_p, sigma_y, mu, Sigma, Sigma_inv, log_Sigma_det, alpha_scale);
        double log_Q_density_given_beta_c = DQPlogPrior_binary(DQP_u_c, quantile_levels, all_quantiles, mu_y_c, sigma_y, mu, Sigma, Sigma_inv, log_Sigma_det, alpha_scale);

        // Log prior density of beta
        double log_beta_prior_p = LogBetaPrior(beta_p, mu_0, Sigma_0_inv, log_Sigma_0_det);
        double log_beta_prior_c = LogBetaPrior(beta_c, mu_0, Sigma_0_inv, log_Sigma_0_det);

        // Log posterior of beta (proposal density ratio is 1 due to symmetry)
        double log_posterior_p = log_like_p + log_beta_prior_p + log_Q_density_given_beta_p;
        double log_posterior_c = log_like_c + log_beta_prior_c + log_Q_density_given_beta_c;

        double log_a_star = log_posterior_p - log_posterior_c;

        // (3) Decide to update
        // [Decide to accept vs. reject]
        double u = R::runif(0, 1);
        if (!Rcpp::NumericVector::is_na(log_a_star)) {
            // consider only when Q_p is reasonably generated
            if (log(u) <= log_a_star) {
                beta_c = beta_p;
                }
            }
    }
    return beta_c;
}