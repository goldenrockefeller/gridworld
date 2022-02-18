from scipy.special import gamma, psi, polygamma, zeta, loggamma
from numpy import log
from scipy import optimize
from numpy import array, zeros, inf, exp, minimum, maximum, sum,  abs, where, isfinite
from numpy.random import dirichlet
from numpy import finfo
import numpy as np


tiny = np.finfo(np.float64).tiny

# cross-entropy and kullback-leibler (kl) divergence are related.
# with the right formulation, minimizing the kl divergence is the same as minimizing the cross-entropy

# loggamma(x) is more stable than log(gamma(x))

#  Optimizing the exponential of the dirichlet parameters is more stable since
# the parameters are constrained to be positive and the exponential is not constrained

def dirichlet_entropy(dist):
    # from https://en.wikipedia.org/wiki/Dirichlet_distribution#Entropy
    return (
        - loggamma(dist.sum())
        + loggamma(dist).sum()
        + (dist.sum()-len(dist)) * psi(dist.sum())
        - ( (dist - 1.) * psi(dist) ).sum()
    )


def dirichlet_divergence_to_dirichlet(ref_dist, dist):
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


def dirichlet_divergence_to_datum(datum, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    return (
        loggamma(dist).sum()
        - loggamma(dist.sum())
        - ((dist - 1.)*log(datum)).sum()
    )

def dirichlet_divergence_to_data(data, dist):
    return sum([dirichlet_divergence_to_datum(datum, dist) for datum in data]) / len(list(data))

def mean_log_of_data(data):
    mean_log_data = 0. * data[0]

    for datum in data:
        mean_log_data += log(datum + tiny)
    mean_log_data /= len(data)

    return mean_log_data

def dirichlet_divergence_to_mean_log_data(mean_log_data, dist):
    """
    Note that dirichlet_divergence_to_data(data, dist) should equal
    dirichlet_divergence_to_mean_log_data(mean_log_of_data(data), dist)
    but the latter is much faster
    """
    return (
        loggamma(dist).sum()
        - loggamma(dist.sum())
        - ((dist - 1.)*mean_log_data).sum()
    )

def scorer(ref_dist, kl_penalty_factor, data):
    def score(dist):
        return dirichlet_divergence_to_data(data, dist) + kl_penalty_factor * dirichlet_divergence_to_dirichlet(ref_dist, dist)
    return score


def fast_scorer(ref_dist, kl_penalty_factor, data):
    mean_log_data = mean_log_of_data(data)

    def score(dist):
        return dirichlet_divergence_to_mean_log_data(mean_log_data, dist) + kl_penalty_factor * dirichlet_divergence_to_dirichlet(ref_dist, dist)
    return score


def exp_scorer(ref_dist, kl_penalty_factor, data):
    def score(ln_dist):
        dist = np.exp(ln_dist)
        return dirichlet_divergence_to_data(data, dist) + kl_penalty_factor * dirichlet_divergence_to_dirichlet(ref_dist, dist)
    return score


def fast_exp_scorer(ref_dist, kl_penalty_factor, data):

    mean_log_data = mean_log_of_data(data)

    def score(ln_dist):
        dist = np.exp(ln_dist)

        return dirichlet_divergence_to_mean_log_data(mean_log_data, dist) + kl_penalty_factor * dirichlet_divergence_to_dirichlet(ref_dist, dist)
    return score


def min_divergence_dirichlet(ref_dist, kl_penalty_factor, data):
    score = fast_scorer(ref_dist, kl_penalty_factor, data)
    result = optimize.minimize(score, ref_dist, method='nelder-mead')
    return result.x

def min_divergence_dirichlet_lbgfs(ref_dist, kl_penalty_factor, data):
    score = fast_scorer(ref_dist, kl_penalty_factor, data)
    jac = jac_fn(ref_dist, kl_penalty_factor, data)
    result = optimize.minimize(score, ref_dist, method='L-BFGS-B',
        jac = jac, options = {"maxcor" : 10})
    return result.x

def min_divergence_dirichlet_cg(ref_dist, kl_penalty_factor, data):
    score = fast_scorer(ref_dist, kl_penalty_factor, data)
    jac = jac_fn(ref_dist, kl_penalty_factor, data)
    result = optimize.minimize(score, ref_dist, method='CG',
        jac = jac)
    return result.x

def min_divergence_dirichlet_exp_lbgfs(ref_dist, kl_penalty_factor, data):
    score = fast_exp_scorer(ref_dist, kl_penalty_factor, data)
    jac = jac_exp_fn(ref_dist, kl_penalty_factor, data)
    result = optimize.minimize(score, np.log(ref_dist), method='L-BFGS-B',
        jac = jac, options = {"maxcor" : 1})
    return np.exp(result.x)

def min_divergence_dirichlet_exp_cg(ref_dist, kl_penalty_factor, data):
    score = fast_exp_scorer(ref_dist, kl_penalty_factor, data)
    jac = jac_exp_fn(ref_dist, kl_penalty_factor, data)
    result = optimize.minimize(score, np.log(ref_dist), method='CG',
        jac = jac)
    return np.exp(result.x)


def jac_fn(ref_dist, kl_penalty_factor, data):
    mean_log_data = 0. * data[0]

    for datum in data:
        mean_log_data += log(datum + tiny)
    mean_log_data /= len(data)

    psi_ref = psi(ref_dist) - psi(ref_dist.sum())

    def jac(dist):
        psi_dist_sum = psi(dist.sum())
        psi_dist = psi(dist)

        return (
            (1 + kl_penalty_factor) * (- psi_dist_sum +  psi_dist)
            - mean_log_data
            - kl_penalty_factor  *psi_ref
        )

    return jac

def jac_exp_fn(ref_dist, kl_penalty_factor, data):
    mean_log_data = 0. * data[0]

    for datum in data:
        mean_log_data += log(datum + tiny)
    mean_log_data /= len(data)

    psi_ref = psi(ref_dist) - psi(ref_dist.sum())

    def jac(ln_dist):
        dist = np.exp(ln_dist)
        psi_dist_sum = psi(dist.sum())
        psi_dist = psi(dist)

        return dist * (
            (1 + kl_penalty_factor) * (- psi_dist_sum +  psi_dist)
            - mean_log_data
            - kl_penalty_factor  *psi_ref
        )

    return jac

