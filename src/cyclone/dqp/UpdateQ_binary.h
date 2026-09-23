#include <iostream>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Last revised: 3/1/2023 

// [[Rcpp::export]]
List proposeQ_chunk(uvec ind, vec Z_c, mat Sigma, double delta, double a1, double a2, vec Q_Lu, vec Q_Ru)
{
    // vector Z_c = normal scale vector related to current quantile vector //
    // vector Q_Lu = left-endpoints of Q given its neighbors (uniform scale) //
    // vector Q_Ru = right-endpoints of Q given its neighbors (uniform scale) //

    // Subsetting Sigma //
    vec rowind = zeros<vec>(Sigma.n_rows);
    vec colind = zeros<vec>(Sigma.n_cols);
    rowind.elem(ind).fill(1);
    colind.elem(ind).fill(1);
    mat Sigma11 = delta * Sigma.submat(ind, ind); 
    mat Sigma12 = delta * Sigma.submat(arma::find(rowind==1), arma::find(colind==0));
    mat Sigma21 = Sigma12.t();
    mat Sigma22 = delta * Sigma.submat(arma::find(rowind==0), arma::find(colind==0));
    mat Sigma22_inv = inv(Sigma22);

    // conditional parameters //
    vec cond_mean =Z_c.elem(ind); // + Sigma12 %*% Sigma22_inv %*% (Z_p_given - mu2) // = always zero
    mat cond_var = Sigma11 - Sigma12 * Sigma22_inv * Sigma21;

    // sample a chunk //
    vec Z_p_ind = sampleZ(cond_mean, cond_var);

    // calculate U, V, Q chunks //
    vec U_p_ind = sampleU(Z_p_ind, cond_mean, cond_var); // element-wise
    vec V_p_ind = sampleV(U_p_ind, a1, a2);
    vec Q_pu_ind = (1-V_p_ind) % Q_Lu.elem(ind) + V_p_ind % Q_Ru.elem(ind);

    return List::create(Named("Q_pu_ind") = Q_pu_ind, Named("Z_p_ind") = Z_p_ind);
}

// [[Rcpp::export]]
double LogQdensity(vec Q_pu, vec Q_Lu, vec Q_Ru, vec Z_p, vec Z_c, mat Sigma, vec mu_y, vec sigma_y,
                   double delta, double a1, double a2, mat Sigma_inv, double log_Sigma_det) {

    vec V_p = (Q_pu - Q_Lu) / (Q_Ru - Q_Lu);
    double logJ = LogJacobianC(Q_pu, Q_Lu, Q_Ru, mu_y, sigma_y, V_p, a1, a2, Z_p, Z_c, delta*Sigma);
    mat Sigma_inv_p = Sigma_inv/delta;
    double log_Sigma_det_p = Z_p.n_elem * log(delta) + log_Sigma_det;
    double logq = logMVNdensity(Z_p, Z_c, Sigma_inv_p, log_Sigma_det_p) + logJ;

    return logq;
}


// [[Rcpp::export]]
mat update_row_part(mat DQP_p, int tau_ind, vec Q_pu, uvec ind){

    rowvec DQP_p_row = DQP_p.row(tau_ind);  // step1. find tau row
    DQP_p_row(ind) = Q_pu(ind);             // step2. replace the corresponding elements
    DQP_p.row(tau_ind) = DQP_p_row;         // step3. update the entire matrix
    return DQP_p;
}


// [[Rcpp::export]]
List UpdateQ(mat DQP_c_y, mat DQP_Z_c, vec beta, vec y, vec mu_y, vec sigma_y, vec group_ind, List quantile_levels, vec all_quantiles,
             vec mu, mat Sigma, mat Sigma_inv, double log_Sigma_det, double delta, double alpha_scale=1, int n_chunk=5)
{
    // [input: mat DQP_c (y scale), all_quantiles] //
    // vec all_quantiles = as_numeric(rownames(DQP_c_rcpp)); // we get all_quantiles from rownames
    
    mat DQP_c(DQP_c_y.n_rows, DQP_c_y.n_cols);
    for (uword r = 0; r < DQP_c_y.n_rows; r++) {
        DQP_c.row(r) = pnorm_cpp(DQP_c_y.row(r).t(), mu_y, sigma_y).t();
    }

    vec quantiles_estimated = {0, 1};
    uvec ind_chunks = chunk_indices(DQP_c.n_cols, n_chunk);

    for (uword i = 0; i < quantile_levels.size(); i++) {
        vec quantile_targets = as<vec>(quantile_levels[i]);

        for (uword j = 0; j < quantile_targets.n_elem; j++) {
            double tau = quantile_targets(j);
            
            // uniform scale
            int tau_ind = arma::as_scalar(find(all_quantiles == tau));
            vec Q_cu = DQP_c.row(tau_ind).t();
            vec Z_c = DQP_Z_c.row(tau_ind).t();
            vec Q_Lu_p = DQP_c.row(tau_ind-1).t();
            vec Q_Ru_p = DQP_c.row(tau_ind+1).t();

            double a1 = 1; double a2 = 1; // propose Q from scaled uniform //
            
            // update chunks (lower-dimensional update) //
            for (uword i = 0; i < ind_chunks.n_elem-1; i++){
                vec Z_p = Z_c;
                vec Q_pu = Q_cu;

                // (i) Propose a new candidate  
                uvec ind = regspace<uvec>(ind_chunks(i), ind_chunks(i+1)-1);
                List res_p = proposeQ_chunk(ind, Z_c, Sigma, delta, a1, a2, Q_Lu_p, Q_Ru_p);
                Q_pu(ind) = as<arma::vec>(res_p["Q_pu_ind"]); 
                Z_p(ind) = as<arma::vec>(res_p["Z_p_ind"]); 
                
                // (ii) Calculate the ratio of conditional proposal density
                // this is a joint density (instead of calculating 1-dim conditional density)
                double LogQdensity_c = LogQdensity(Q_cu, Q_Lu_p, Q_Ru_p, Z_c, Z_p, Sigma, mu_y, sigma_y, delta, a1, a2, Sigma_inv, log_Sigma_det);
                double LogQdensity_p = LogQdensity(Q_pu, Q_Lu_p, Q_Ru_p, Z_p, Z_c, Sigma, mu_y, sigma_y, delta, a1, a2, Sigma_inv, log_Sigma_det);

                // (iii) Calculate the (log)prior densities
                mat DQP_p = update_row_part(DQP_c, tau_ind, Q_pu, ind);

                double logprior_c = DQPlogPrior_binary(DQP_c, quantile_levels, all_quantiles, mu_y, sigma_y, mu, Sigma, Sigma_inv, log_Sigma_det, alpha_scale);
                double logprior_p = DQPlogPrior_binary(DQP_p, quantile_levels, all_quantiles, mu_y, sigma_y, mu, Sigma, Sigma_inv, log_Sigma_det, alpha_scale);

                // (iv) Calculate the (log)likelihoods
                double loglike_c = LogLikeC(DQP_c, all_quantiles, y, mu_y, sigma_y, group_ind);
                double loglike_p = LogLikeC(DQP_p, all_quantiles, y, mu_y, sigma_y, group_ind);

                // (v) Calculate the (log) acceptance probability
                double log_a_star = (loglike_p + logprior_p) - (loglike_c + logprior_c) + (LogQdensity_c - LogQdensity_p);

                // [Decide to accept vs. reject]
                double u = R::runif(0, 1);
                if (!Rcpp::NumericVector::is_na(log_a_star)) {
                    // consider only when Q_p is reasonably generated
                    if (log(u) <= log_a_star) {
                        DQP_c = DQP_p;
                        DQP_Z_c.row(tau_ind) = Z_p.t();
                        Q_cu(ind) = Q_pu(ind); 
                        Z_c(ind) = Z_p(ind); 
                    }
                }
            }
        }
        quantiles_estimated = sort(join_cols(quantiles_estimated, quantile_targets));
    }

    // transform back to y scale //
    for (uword r = 0; r < DQP_c.n_rows; r++) {
        DQP_c_y.row(r) = qnorm_cpp(DQP_c.row(r).t(), mu_y, sigma_y).t();
    }

    // [output Q =  arma::mat, Z = arma::mat]
    return List::create(Named("Q") = DQP_c_y, Named("Z") = DQP_Z_c);
}
