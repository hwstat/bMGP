# Vendored from the qmp package of Edwin Fong, https://github.com/edfong/qmp
# (file qmp/qmp_reg_functions.py), the reference implementation of the
# quantile martingale posterior of Fong and Yiu. Copied unchanged; used by
# src/cyclone/ for the cyclone runtime and posterior comparison.
#
# MIT License, Copyright (c) 2024 Edwin Fong. The full notice is in
# src/quantile/qmp/LICENSE.

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp,betainc
from jax.scipy.stats import norm,t
from jax.lax import fori_loop,scan,cond
from functools import partial
from jax.random import PRNGKey,split,permutation

from .utils.bivariate_copula import log_Huv,ndtri_, norm_logbicop_approx
from .qmp_functions import rearrange_Q,rearrange_Q_B


#with rearrangement
@jit
def update_beta(carry,i):
    beta_plot,preq_score,u_plot,y,x,a,b,k,du = carry

    #Compute prequential log score
    Q_plot_x = jnp.sum(x[i].reshape(1,-1)*beta_plot,axis = 1)
    v = jnp.mean(Q_plot_x <= y[i])

    Q_plot_x = rearrange_Q(Q_plot_x) # Rearrange conditional quantile to differentiate
    q_plot = jnp.diff(Q_plot_x)/du
    preq_score = preq_score - jnp.log(jnp.interp(v, u_plot[1:],q_plot)) #logscore - linear interpolate for quantile density
    

    #Update quantile 
    alpha = a/((i+2))
    rho = jnp.sqrt(1-b/((i+1)**(k)))
    beta_plot = beta_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,v,rho))).reshape(-1,1)*(x[i].reshape(1,-1))

    carry = beta_plot,preq_score,u_plot,y,x,a,b,k,du
    return carry,i

#Scan over y_{1:n}
@jit
def update_beta_scan(carry,rng):
    return scan(update_beta,carry,rng)

#Run loop and fit
@jit
def fit_beta(hyperparam,y,x,du = 0.005):
    #Set initial Q0
    n = jnp.shape(y)[0] 
    d = jnp.shape(x)[1]
    u_plot = jnp.arange(du, 1, du)
    n_plot = jnp.shape(u_plot)[0]
    
    beta_plot = jnp.zeros((n_plot,d)) #Initial coefficients equal to 0
    # Set intercept to interpolate lower and upper quartile
    q25 = jnp.quantile(y,0.25)
    q75 = jnp.quantile(y,0.75)
    b = 2*(q75 - q25) #slope = 2*iqr
    c = q25 - 0.25*b
    beta_plot = beta_plot.at[:,0].set(b*u_plot + c) #set it to be linear on interquartile range
    
    #Initialize 
    preq_score = 0
    a = hyperparam[0]
    b = hyperparam[1]
    k = hyperparam[2]
    rng = jnp.arange(n)

    #Fit model
    carry = beta_plot,preq_score,u_plot,y,x,a,b,k,du
    carry,rng = update_beta_scan(carry,rng)
    beta_plot,preq_score,*_= carry
    preq_score = preq_score/n
    return beta_plot,preq_score

@partial(jit, static_argnums = (3,4))
def fit_beta_perm(hyperparam,y,x,du = 0.005,n_perm = 10,seed = 100):
    #Define vmap function
    fit_beta_perm_tmp = jit(vmap(fit_beta,(None,0,0)))
    d = np.shape(x)[1]

    #Generate random permutations
    key = PRNGKey(seed)
    key,*subkey = split(key,n_perm +1)
    subkey = jnp.array(subkey)
    y_perm = vmap(permutation,(0,None))(subkey,y).reshape(n_perm,-1,1)
    x_perm = vmap(permutation,(0,None))(subkey,x).reshape(n_perm,-1,d) #fix this to match


    #Fit
    beta_plot_perm,preq_score_perm = fit_beta_perm_tmp(hyperparam,y_perm, x_perm)
    beta_plot = jnp.mean(beta_plot_perm,axis = 0)
    preq_score = jnp.mean(preq_score_perm,axis = 0)
    return beta_plot, preq_score

# Computing conditional quantiles
# beta = n_plot x d, x_plot = n_plot_x x d
rearrange_Q_x = vmap(rearrange_Q,(1))
@jit
def compute_Qx(beta,x_plot):
    Q_x = jnp.dot(beta,jnp.transpose(x_plot)) #n_plot x n_plot_x
    Q_x = rearrange_Q_x(Q_x)
    return(Q_x)


compute_Qx_B = vmap(compute_Qx,(0,None))