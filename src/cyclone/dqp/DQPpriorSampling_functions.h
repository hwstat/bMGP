#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// last revised: 3/1/2023

// [[Rcpp::export]]
List DQPbinarySampling(vec mu, mat Sigma, List quantile_levels, double alpha_scale = 1) 
{
    int n = mu.size();

    vec Q_left = zeros<vec>(n);
    vec Q_right = ones<vec>(n);
    mat Q_mat = join_rows(Q_left, Q_right).t();
    mat Z_mat = join_rows(Q_left, Q_right).t(); // just a placeholder

    vec quantiles_estimated = {0, 1};

    int quantile_levels_length = quantile_levels.size();
    for(int i = 0; i < quantile_levels_length; i++) {
        vec quantile_targets = as<vec>(quantile_levels[i]);

        int quantile_targets_length = quantile_targets.size();
        for(int j = 0; j < quantile_targets_length; j++) {
            double tau_l = quantiles_estimated[j];
            double tau_r = quantiles_estimated[j+1];
            
            vec Q_left = Q_mat.row(2*j).t(); // row vector
            vec Q_right = Q_mat.row(2*j+1).t(); // row vector
            
            double tau = quantile_targets[j];
            double a1 = (tau - tau_l) * pow(i + alpha_scale, 2);
            double a2 = (tau_r - tau) * pow(i + alpha_scale, 2);

            vec Z = sampleZ(mu, Sigma);
            vec U = sampleU(Z, mu, Sigma);
            vec V = sampleV(U, a1, a2);
            vec Q = sampleQ(Q_left, Q_right, V); // column vector (default)

            int nr = Q_mat.n_rows;
            Q_mat = join_cols(Q_mat.rows(0,2*j), Q.t(), Q_mat.rows(2*j+1,nr-1));
            Z_mat = join_cols(Z_mat.rows(0,2*j), Z.t(), Z_mat.rows(2*j+1,nr-1));
        }
        quantiles_estimated = sort(join_cols(quantiles_estimated, quantile_targets));
    }
    NumericMatrix Q_mat_rcpp = clone(wrap(Q_mat));
    CharacterVector Q_names = clone(wrap(quantiles_estimated));
    rownames(Q_mat_rcpp) = Q_names;

    List outputList = List::create(Named("Q") = Q_mat_rcpp, Named("Z") = Z_mat);

    return outputList;
}
