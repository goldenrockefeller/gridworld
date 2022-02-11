from scipy.special import gamma, psi, polygamma, zeta, loggamma
from numpy import log
from scipy import optimize
from numpy import array, zeros, inf, exp, minimum, maximum, sum,  abs, where, isfinite
from numpy.random import dirichlet
from numpy import finfo
import numpy as np


tiny = np.finfo(np.float64).tiny



def dist_entropy(dist):
    return (
        - loggamma(dist.sum())
        + loggamma(dist).sum()
        + (dist.sum()-len(dist)) * psi(dist.sum())
        - ( (dist - 1.) * psi(dist) ).sum()
    )


def dist_to_dist_entropy(ref_dist, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    ref_dist_sum = ref_dist.sum()
    dist_sum = dist.sum()

    if not np.isfinite(loggamma(ref_dist_sum)).all():
        raise ValueError("loggamma(ref_dist_sum)", ref_dist_sum)

    return (
        loggamma(ref_dist_sum)
        - loggamma(ref_dist).sum()
        - loggamma(dist_sum)
        + loggamma(dist).sum()
        + ( (ref_dist - dist) * (psi(ref_dist) - psi(ref_dist_sum)) ).sum()
    )


def data_to_dist_entropy(data, dist):
    return sum([datum_to_dist_entropy(datum, dist) for datum in data]) / len(list(data))

def data_to_dist_entropy_pk(log_pk, dist):
    return (
        loggamma(dist).sum()
        - loggamma(dist.sum())
        - ((dist - 1.)*log_pk).sum()
    )

def datum_to_dist_entropy(datum, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    return (
        loggamma(dist).sum()
        - loggamma(dist.sum())
        - ((dist - 1.)*log(datum)).sum()
    )


def scorer(ref_dist, kl_penalty_factor, data):
    def score(dist):
        return data_to_dist_entropy(data, dist) + kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score

def scorer_2(ref_dist, kl_penalty_factor, dest):
    def score(dist):
        return dist_to_dist_entropy(dest, dist)+ kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score


def scorer_pk(ref_dist, kl_penalty_factor, data):
    log_pk = 0. * data[0]

    for datum in data:
        log_pk += log(datum + tiny)
    log_pk /= len(data)

    def score(dist):
        return data_to_dist_entropy_pk(log_pk, dist) + kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score

def exp_scorer(ref_dist, kl_penalty_factor, data):
    def score(ln_dist):
        dist = np.exp(ln_dist)
        return data_to_dist_entropy(data, dist) + kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score


def exp_scorer_pk(ref_dist, kl_penalty_factor, data):
    log_pk = 0. * data[0]

    for datum in data:
        log_pk += log(datum + tiny)
    log_pk /= len(data)

    def score(ln_dist):
        dist = np.exp(ln_dist)

        return data_to_dist_entropy_pk(log_pk, dist) + kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score


def min_entropy_dist(ref_dist, kl_penalty_factor, data):
    score = scorer_pk(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, ref_dist, method='nelder-mead')

def min_entropy_dist_lbgfs(ref_dist, kl_penalty_factor, data):
    score = scorer_pk(ref_dist, kl_penalty_factor, data)
    jac = jac_fn(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, ref_dist, method='L-BFGS-B',
        jac = jac, options = {"maxcor" : 10})

def min_entropy_dist_cg(ref_dist, kl_penalty_factor, data):
    score = scorer_pk(ref_dist, kl_penalty_factor, data)
    jac = jac_fn(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, ref_dist, method='CG',
        jac = jac)

def min_entropy_dist_exp_lbgfs(ref_dist, kl_penalty_factor, data):
    score = exp_scorer_pk(ref_dist, kl_penalty_factor, data)
    jac = jac_exp_fn(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, np.log(ref_dist), method='L-BFGS-B',
        jac = jac, options = {"maxcor" : 1})

def min_entropy_dist_exp_cg(ref_dist, kl_penalty_factor, data):
    score = exp_scorer_pk(ref_dist, kl_penalty_factor, data)
    jac = jac_exp_fn(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, np.log(ref_dist), method='CG',
        jac = jac)


def jac_fn(ref_dist, kl_penalty_factor, data):
    log_pk = 0. * data[0]

    for datum in data:
        log_pk += log(datum + tiny)
    log_pk /= len(data)

    psi_ref = psi(ref_dist) - psi(ref_dist.sum())

    def jac(dist):
        psi_dist_sum = psi(dist.sum())
        psi_dist = psi(dist)

        return (
            (1 + kl_penalty_factor) * (- psi_dist_sum +  psi_dist)
            - log_pk
            - kl_penalty_factor  *psi_ref
        )

    return jac

def jac_exp_fn(ref_dist, kl_penalty_factor, data):
    log_pk = 0. * data[0]

    for datum in data:
        log_pk += log(datum + tiny)
    log_pk /= len(data)

    psi_ref = psi(ref_dist) - psi(ref_dist.sum())

    def jac(ln_dist):
        dist = np.exp(ln_dist)
        psi_dist_sum = psi(dist.sum())
        psi_dist = psi(dist)

        return dist * (
            (1 + kl_penalty_factor) * (- psi_dist_sum +  psi_dist)
            - log_pk
            - kl_penalty_factor  *psi_ref
        )

    return jac

def min_entropy_dist_2(ref_dist, kl_penalty_factor, dest):
    score = scorer_2(ref_dist, kl_penalty_factor, dest)
    return optimize.minimize(score, ref_dist, method='nelder-mead')