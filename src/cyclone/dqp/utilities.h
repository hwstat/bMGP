#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
vec as_numeric(CharacterVector v)
{
    int n = v.size();
    vec out(n);
    std::string s;
    for(int i=0; i<n; i++){
        s = std::string(v[i]);
        out(i) = std::stod(s);
    }
    return out;
}

// [[Rcpp::export]]
double normalCDF(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}

// [[Rcpp::export]]
vec sampleZ(vec mu, mat Sigma)
{
    Function chol("chol");

    /*Generate W vector from standard normal distribution*/
    int n_mu = mu.size();
    vec W = Rcpp::rnorm(n_mu); 

    /*Find A using Cholesky decomposition*/ 
    mat A = as<mat>(chol(Sigma)); 
    
    /*Obtain X using affine transformation*/
    vec Z = mu + A * W;

    return Z;
}


// [[Rcpp::export]]
vec sampleU(vec Z, vec mu, mat Sigma)
{
    int n_mu = mu.size();
    vec U(n_mu);
    for (int i=0; i<n_mu; ++i){
        U[i] = normalCDF((Z[i]-mu[i])/sqrt(Sigma(i,i)));
    }
    return U;
}


// [[Rcpp::export]]
vec sampleV(vec U, double a1, double a2)
{
    int n = U.size();
    vec V(n);
    for (int i=0; i<n; i++) {
        V[i] = R::qbeta(U[i], a1, a2, true, false);
    }
    return V;
}

// [[Rcpp::export]]
vec sampleQ(vec Q_left, vec Q_right, vec V)
{
    vec Q = Q_left % (1-V) + Q_right % V; // elementwise multiplication
    return Q;
}

// [[Rcpp::export]]
double logMVNdensity(vec Z, vec mu, mat Sigma_inv, double log_Sigma_det)
{
    int n_mu = mu.size();
    vec diff = Z - mu;
    double quad_form = dot(diff, Sigma_inv * diff);
    double log_density = -0.5 * n_mu * log(2 * M_PI) - 0.5 * log_Sigma_det - 0.5 * quad_form;
    return log_density;
}

// [[Rcpp::export]]
vec pnorm_cpp(const vec& x, const vec& mu, const vec& sigma) {
    // Define a function to calculate pnorm in C++.
    // This is not strictly necessary, but it can be faster than using R's pnorm.
    vec z = (x - mu) / (sigma * sqrt(2));
    vec p = 0.5 * (1 + erf(z));
    return p;
}

// [[Rcpp::export]]
vec qnorm_cpp(const vec& x, const vec& mu, const vec& sigma) {
    // Define a function to calculate pnorm in C++.
    // This is not strictly necessary, but it can be faster than using R's pnorm.
    vec z(x.n_elem);
    for(int i=0; i<x.n_elem; i++){
        z(i) = R::qnorm(x(i), mu(i), sigma(i), true, false);
    }
    return z;
}

// [[Rcpp::export]]
vec recoverV(vec Qu, vec Q_Lu, vec Q_Ru) {
    // Define a function to recover V.
    // This function assumes that Qu, Q_Lu, and Q_Ru are row vectors.
    // If they are matrices, the function needs to be modified.
    return (Qu - Q_Lu) / (Q_Ru - Q_Lu);
}

// [[Rcpp::export]]
vec hC(vec V, double a, double b, vec mu, mat Sigma)
{
    int n = V.size();
    vec Z(n);
    vec sig = sqrt(diagvec(Sigma));
    for(int i=0; i<n; i++){
        double U = R::pbeta(V[i], a, b, true, false);
        Z[i] = R::qnorm(U, mu[i], sig[i], true, false);
    }
    return Z;
}

// [[Rcpp::export]]
double LogJacobianC(vec Qu, vec Q_Lu, vec Q_Ru, vec mu_y, vec sigma_y, 
vec V, double a, double b, vec Z, vec mu, mat Sigma)
{
    int n = Qu.size();
    vec log_U_portion(n), log_V_portion(n), log_Q_portion(n), log_Q_y_portion(n);
    vec sig = sqrt(diagvec(Sigma));

    for(uword i=0; i<n; i++){
        log_U_portion[i] = -R::dnorm(Z[i], mu[i], sig[i], true); // log scale, portion from transforming Z -> U
        log_V_portion[i] = R::dbeta(V[i], a, b, true); // log scale, portion from transforming U -> V
        log_Q_portion[i] = -log(Q_Ru[i] - Q_Lu[i]); // log scale, portion from transforming V -> Q
        log_Q_y_portion[i] = R::dnorm(R::qnorm(Qu[i], mu_y[i], sigma_y[i], true, false), mu_y[i], sigma_y[i], true); // log scale, portion from transforming Q -> Q_y (y-scale)
    }
    double logJ = sum(log_U_portion) + sum(log_V_portion) + sum(log_Q_portion) + sum(log_Q_y_portion);
    return logJ;
}

// [[Rcpp::export]]
uvec chunk_indices(int n, int n_chunk) {
  
  int chunk_size = std::ceil(n / static_cast<double>(n_chunk));
  
  uvec indices;
  indices.zeros(n_chunk + 1);
  
  for (int i = 0; i < n_chunk; ++i) {
    int start_idx = i * chunk_size;
    int end_idx = std::min((i + 1) * chunk_size, n);
    indices(i + 1) = end_idx;
    if (start_idx >= end_idx) {
      Rcpp::stop("Invalid number of chunks specified");
    }
  }
  return indices;
}