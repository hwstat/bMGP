import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap,jacfwd,jacrev,random,remat,value_and_grad
from jax.scipy.special import ndtri,erfc,logsumexp,betainc
from jax.scipy.stats import norm,t
from jax.lax import fori_loop,scan,cond
from functools import partial
from jax.random import PRNGKey,split,permutation

from .utils.bivariate_copula import log_Huv,ndtri_, norm_logbicop_approx

@jit
def update_Q(carry,i):
    Q_plot,preq_score,n_rearr,u_plot,y,a,c,k,du = carry

    v = jnp.mean(Q_plot <= y[i])
    q_plot = jnp.diff(Q_plot)/du
    preq_score = preq_score - jnp.log(jnp.interp(v, u_plot[1:],q_plot)) #logscore - linear interpolate for quantile density

    alpha = a/((i+2))
    rho = jnp.sqrt(1-c/((i+1)**(k)))
    Q_plot = Q_plot + alpha*(u_plot - jnp.exp(log_Huv(u_plot,v,rho)))

    ind_rearrange = jnp.any(jnp.diff(Q_plot)<0)
    n_rearr = n_rearr + ind_rearrange
    Q_plot = cond(ind_rearrange,lambda x:  jnp.sort(x), lambda x: x ,Q_plot) #this works because the grid is the same
    carry = Q_plot,preq_score,n_rearr,u_plot,y,a,c,k,du
    return carry,i

@jit
def update_Q_scan(carry,rng):
    return scan(update_Q,carry,rng)

@jit
def fit_Q(hyperparam,y,du = 0.005):
    n = jnp.shape(y)[0]
    u_plot = jnp.arange(du, 1, du)
    Q_plot =  (jnp.min(y)) + (jnp.ptp(y))*u_plot #Uniform on range

    preq_score = 0
    n_rearr= 0
    a = hyperparam[0]
    c = hyperparam[1]
    k = hyperparam[2]
    rng = jnp.arange(n)

    carry = Q_plot,preq_score,n_rearr,u_plot,y,a,c,k,du
    carry,rng = update_Q_scan(carry,rng)
    Q_plot,preq_score,n_rearr,*_= carry
    preq_score = preq_score/n
    return Q_plot,preq_score,n_rearr

@partial(jit,static_argnums = (2,3))
def fit_Q_perm(hyperparam,y,du = 0.005,n_perm = 10,seed = 100):
    fit_Q_perm_tmp = jit(vmap(fit_Q,(None,0,)))

    key = PRNGKey(seed)
    key,*subkey = split(key,n_perm +1)
    subkey = jnp.array(subkey)
    y_perm = vmap(permutation,(0,None))(subkey,y).reshape(n_perm,-1,1)

    Q_plot_perm,preq_score_perm,n_rearr_perm = fit_Q_perm_tmp(hyperparam,y_perm)
    Q_plot = jnp.mean(Q_plot_perm,axis = 0)
    preq_score = jnp.mean(preq_score_perm,axis = 0)
    n_rearr = jnp.mean(n_rearr_perm)
    return Q_plot, preq_score, n_rearr

@jit
def rearrange_Q(Q_plot):
    ind_rearrange = jnp.any(jnp.diff(Q_plot)<0)
    Q_plot = cond(ind_rearrange,lambda x:  jnp.sort(x), lambda x: x ,Q_plot) #this works because the grid is the same
    return Q_plot

rearrange_Q_B = vmap(rearrange_Q, (0))
