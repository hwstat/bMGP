import numpy as np
import scipy as sp
from functools import partial

import jax.numpy as jnp
from jax import grad,value_and_grad, jit, vmap,jacfwd,jacrev,random
from jax.scipy.stats import norm
from jax.lax import fori_loop,scan

from . import qmp_functions as qmp
from .utils.bivariate_copula import ndtri_

from .utils.bivariate_copula import log_Huv,ndtri_

@jit
def update_Q_norearr(carry,i):
    Q_plot,u_plot,vT,a,c,k = carry

    alpha = a/((i+2))
    rho = jnp.sqrt(1-c/((i+1)**(k)))
    Q_plot = Q_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,vT[i],rho)))

    carry = Q_plot,u_plot,vT,a,c,k
    return carry,i

@jit
def update_Q_norearr_scan(carry,rng):
    return scan(update_Q_norearr,carry,rng)

@partial(jit,static_argnums = (5,6))
def PR_loop(key,Q_init,a,c,k,n,T,du = 0.005):

    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,))

    vT = jnp.append(jnp.zeros((n)),a_rand)

    u_plot = jnp.arange(du, 1, du)

    inputs = Q_init,u_plot,vT,a,c,k
    rng = jnp.arange(n,n+T)
    outputs,rng = update_Q_norearr_scan(inputs,rng)
    Q_plot,*_ = outputs

    return Q_plot

PR_loop_B =jit(vmap(PR_loop,(0,None,None,None,None,None,None)),static_argnums = (2,3,4,5,6)) #vmap across B posterior samples

@jit
def update_Q_norearr_conv(carry,i):
    Q_plot,u_plot,vT,a,c,k, Q_init,Q_diff = carry

    alpha = a/((i+2))
    rho =  jnp.sqrt(1-c/((i+1)**(k)))
    Q_plot = Q_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,vT[i],rho)))

    Q_diff = Q_diff.at[i].set(jnp.mean(jnp.abs(Q_plot- Q_init)**2)) #mean L2 difference

    carry =  Q_plot,u_plot,vT,a,c,k, Q_init,Q_diff
    return carry,i

@jit
def update_Q_norearr_scan_conv(carry,rng):
    return scan(update_Q_norearr_conv,carry,rng)

@partial(jit,static_argnums = (5,6))
def PR_loop_conv(key,Q_init,a,c,k,n,T,du = 0.005):

    key, subkey = random.split(key) #split key
    a_rand = random.uniform(subkey,shape = (T,))

    vT = jnp.append(jnp.zeros((n)),a_rand)

    u_plot = jnp.arange(du, 1, du)

    Q_diff = jnp.zeros(n+T)
    inputs = Q_init,u_plot,vT,a,c,k, Q_init, Q_diff
    rng = jnp.arange(n,n+T)
    outputs,rng = update_Q_norearr_scan_conv(inputs,rng)
    Q_plot,u_plot,vT,a,c,k, Q_init,Q_diff = outputs

    return Q_plot, Q_diff

PR_loop_B_conv =jit(vmap(PR_loop_conv,(0,None,None,None,None,None,None)),static_argnums = (2,3,4,5,6)) #vmap across B posterior samples

def approx_PR_B(seed,Q_init,a,c,k,n,B,du = 0.005):
    np.random.seed(seed)

    u_plot = jnp.arange(du, 1, du)
    n_plot = np.shape(u_plot)[0]

    rho_end = np.sqrt(1 - c*(n+1)**(-k))
    cov = np.array([[1,rho_end**2],[rho_end**2,1]])

    zu,zv = np.meshgrid(sp.stats.norm.ppf(u_plot),sp.stats.norm.ppf(u_plot))
    z_plot = np.vstack([zu.ravel(), zv.ravel()]).transpose()
    cop = sp.stats.multivariate_normal.cdf(z_plot,cov = cov).reshape(n_plot,n_plot)
    Sigma = (cop - np.outer(u_plot,u_plot) + 5e-7*np.eye(n_plot))

    chol = np.linalg.cholesky(Sigma)

    gp_samp_smooth = np.transpose(np.dot(chol, np.random.randn(n_plot,B)))

    Q_gp_smooth = Q_init.reshape(1,-1) + a*gp_samp_smooth/np.sqrt(n+1)
    return Q_gp_smooth
