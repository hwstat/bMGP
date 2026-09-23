# Vendored from the qmp package of Edwin Fong, https://github.com/edfong/qmp
# (file qmp/sample_qmp_reg_functions.py), the reference implementation of the
# quantile martingale posterior of Fong and Yiu. Copied unchanged; used by
# src/cyclone/ for the cyclone runtime and posterior comparison.
#
# MIT License, Copyright (c) 2024 Edwin Fong. The full notice is in
# src/quantile/qmp/LICENSE.

import numpy as np
import scipy as sp
from functools import partial

#import jax functions
import jax.numpy as jnp
from jax import grad,value_and_grad, jit, vmap,jacfwd,jacrev,random
from jax.scipy.stats import norm
from jax.lax import fori_loop,scan
from jax.random import dirichlet, choice

#import package functions
from . import qmp_functions as qmp
from .utils.bivariate_copula import ndtri_
from .utils.bivariate_copula import log_Huv,ndtri_

### Predictive resampling functions ###

#### Main function ####
#updating without rearrangement nor prequential score
@jit
def update_beta_norearr(carry,i):
    beta_plot,u_plot,x,ind_xT,vT,a,c,k = carry

    #Update quantile 
    alpha = a/((i+2))
    rho = jnp.sqrt(1-c/((i+1)**(k)))
    #rho = k
    beta_plot = beta_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,vT[i],rho))).reshape(-1,1)*(x[ind_xT[i]].reshape(1,-1))

    carry = beta_plot,u_plot,x,ind_xT,vT,a,c,k
    return carry,i

#Scan over y_{1:n}
@jit
def update_beta_norearr_scan(carry,rng):
    return scan(update_beta_norearr,carry,rng)


# Loop through forward sampling; generate uniform random variables, then use p(y) update from mvcd
@partial(jit,static_argnums = (3,4,5,6,7))
def PR_reg_loop(key,beta_init,x,a,c,k,n,T,du = 0.005):

    #generate uniform random numbers
    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,))

    #Append a_rand to empty vn (for correct array size)
    vT = jnp.append(jnp.zeros((n)),a_rand)

    #Initialize grid
    u_plot = jnp.arange(du, 1, du)

    #Sample x values from BB (to-do)
    key, subkey = random.split(key) #split key
    w = dirichlet(subkey, alpha = jnp.ones(n))
    ind_xT = random.choice(key,a=n,shape = (n+T,),p = w)

    #run forward loop
    inputs = beta_init,u_plot,x,ind_xT,vT,a,c,k
    rng = jnp.arange(n,n+T)
    outputs,rng = update_beta_norearr_scan(inputs,rng)
    beta_plot,*_ = outputs

    return beta_plot

## Vmap over multiple test points, then over multiple seeds
PR_reg_loop_B =jit(vmap(PR_reg_loop,(0,None,None,None,None,None,None,None)),static_argnums = (3,4,5,6,7)) #vmap across B posterior samples
#### ####

### Approximate Sampling ###
# Utility function in Jax for dealing with covariate structure
@jit
def cov_x_GP(w,gp_samp_smooth_indep,x):
    cov_x = jnp.dot(jnp.transpose(x),x*w.reshape(-1,1))
    chol_x = jnp.linalg.cholesky(cov_x)
    return(jnp.dot(gp_samp_smooth_indep, chol_x))
cov_x_GP_B = vmap(cov_x_GP, (0,0,None))

dot_B = vmap(jnp.dot,(None,0))

# Not using JAX as regular scipy has fast multivariate normal
def approx_PR_reg_B(seed,beta_init,x,a,c,k,n,B,du = 0.005):
    np.random.seed(seed)
    d = np.shape(x)[1]

    #Initialize grid
    u_plot = jnp.arange(du, 1, du)
    n_plot = np.shape(u_plot)[0]

    #Compute bandwidth
    rho_end = np.sqrt(1 - c*(n+1)**(-k))
    cov = np.array([[1,rho_end**2],[rho_end**2,1]])

    #Compute covariance matrix
    zu,zv = np.meshgrid(sp.stats.norm.ppf(u_plot),sp.stats.norm.ppf(u_plot))
    z_plot = np.vstack([zu.ravel(), zv.ravel()]).transpose()
    cop = sp.stats.multivariate_normal.cdf(z_plot,cov = cov).reshape(n_plot,n_plot)
    Sigma = (cop - np.outer(u_plot,u_plot) + 5e-7*np.eye(n_plot))
    
    #Cholesky decomp for sampling
    chol = np.linalg.cholesky(Sigma)

    #Sample GP
    gp_samp_smooth_indep = dot_B(chol,np.random.randn(B,n_plot,d))
    
    #Dirichlet weights
    w = np.random.dirichlet(alpha = np.ones(n),size = B)

    #Dot product to get covariate structure
    gp_samp_smooth = cov_x_GP_B(w,gp_samp_smooth_indep,x)

    #Add to beta_init
    beta_gp_smooth = beta_init.reshape(1,-1,d) + a*gp_samp_smooth/np.sqrt(n+1)
    return beta_gp_smooth
