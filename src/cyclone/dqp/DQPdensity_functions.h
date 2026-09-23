#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// last revised: 3/1/2023

// [[Rcpp::export]]
double DQPlogPrior_binary(mat DQP, List quantile_levels, vec all_quantiles, vec mu_y, vec sigma_y, 
vec mu, mat Sigma, mat Sigma_inv, double log_Sigma_det, double alpha_scale = 1) 
{
    // [input: mat DQP (uniform scale), vec all_quantiles] //

    vec quantiles_estimated = {0, 1};

    double out = 0;
    for (uword i = 0; i < quantile_levels.size(); i++) {
        vec quantile_targets = as<vec>(quantile_levels[i]);

        for (uword j = 0; j < quantile_targets.n_elem; j++) {
            double tau = quantile_targets(j);
            double tau_l = quantiles_estimated(j);
            double tau_r = quantiles_estimated(j + 1);
            
            vec Qu = DQP.row(arma::as_scalar(find(all_quantiles == tau))).t();
            vec Q_Lu = DQP.row(arma::as_scalar(find(all_quantiles == tau_l))).t();
            vec Q_Ru = DQP.row(arma::as_scalar(find(all_quantiles == tau_r))).t();
            vec V = recoverV(Qu, Q_Lu, Q_Ru);

            double a = (tau - tau_l) * pow(i + 1 + alpha_scale, 2); // Pyramid level = (i+1)
            double b = (tau_r - tau) * pow(i + 1 + alpha_scale, 2); // Pyramid level = (i+1)
            vec Z = hC(V, a, b, mu, Sigma);

            double logJ = LogJacobianC(Qu, Q_Lu, Q_Ru, mu_y, sigma_y, V, a, b, Z, mu, Sigma);
            double log_mvn_density =  logMVNdensity(Z, mu, Sigma_inv, log_Sigma_det);
            
            out += log_mvn_density + logJ;
        }
        quantiles_estimated = sort(join_cols(quantiles_estimated, quantile_targets));
    }
    return out;
}


