from scipy.special import gamma, psi, polygamma, zeta
from numpy import log
from scipy import optimize
from numpy import array, zeros, inf, exp, minimum, maximum, sum,  abs, where, isfinite
from numpy.random import dirichlet
from numpy import finfo
import numpy as np


tiny = np.finfo(np.float64).tiny



def dist_entropy(dist):
    return (
        - log(gamma(dist.sum()))
        + log(gamma(dist)).sum()
        + (dist.sum()-len(dist)) * psi(dist.sum())
        - ( (dist - 1.) * psi(dist) ).sum()
    )


def dist_to_dist_entropy(ref_dist, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    ref_dist_sum = ref_dist.sum()
    dist_sum = dist.sum()

    if not np.isfinite(log(gamma(ref_dist_sum))).all():
        print("hi")

    return (
        log(gamma(ref_dist_sum))
        - log(gamma(ref_dist)).sum()
        - log(gamma(dist_sum))
        + log(gamma(dist)).sum()
        + ( (ref_dist - dist) * (psi(ref_dist) - psi(ref_dist_sum)) ).sum()
    )


def data_to_dist_entropy(data, dist):
    return sum([datum_to_dist_entropy(datum, dist) for datum in data]) / len(list(data))

def data_to_dist_entropy_pk(log_pk, dist):
    return (
        log(gamma(dist)).sum()
        - log(gamma(dist.sum()))
        - ((dist - 1.)*log_pk).sum()
    )

def datum_to_dist_entropy(datum, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    return (
        log(gamma(dist)).sum()
        - log(gamma(dist.sum()))
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
#
# def h_zeta_2(x):
#     y = 0.
#     prev_y = -inf
#     n = 0.
#
#     while prev_y != y:
#         prev_y = y
#         y += 1. / ((n + x) * (n + x))
#         n += 1.
#
#     return y



# def trigamma(x):
#     # From https://en.wikipedia.org/wiki/Polygamma_function
#     return zeta(2, x)
# #
# def inv_psi(x):
#     s = x
#     prev_s = -inf
#     prev_s_mid = -inf
#
#     s_mid = -inf
#     s_hi = inf
#     s_lo = -inf
#     for i in range(1000):
#         prev_s = s
#         f = psi(exp(s)) - x
#         s = s - (f / (trigamma( exp(s)) * exp(s)))
#
#
#
#         f = psi(exp(s)) - x
#
#         if isfinite(s):
#             if (prev_s == s):
#                 return(exp(s))
#
#             if f == 0:
#                 return exp(s)
#
#
#
#         if f > 0:
#             s_hi = min(s, s_hi)
#             s = s_hi
#         else:
#             s_lo = max(s, s_lo)
#             s = s_lo
#
#
#         prev_s_mid = s_mid
#         s_mid = 0.5 * (s_hi + s_lo)
#
#         if isfinite(s_mid) :
#             f = psi(exp(s_mid)) - x
#
#             if f == 0:
#                 return exp(s_mid)
#
#             if f > 0:
#                 s_hi = min(s_mid, s_hi)
#                 s_mid = s_hi
#             else:
#                 s_lo = max(s_mid, s_lo)
#                 s_mid = s_lo
#
#             if (prev_s_mid == s_mid):
#                 return(exp(s_mid))
#
#
#     return exp(s)
#
#
#
#
#
#
# def my_optimize(ref_dist, kl_penalty_factor, data):
#     # Using fixed point method http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
#     dist = ref_dist.copy()
#
#     log_pk = 0. * data[0]
#     for datum in data:
#         log_pk += log(datum + 1e-9)
#     log_pk /= len(data)
#
#     log_pk = log_pk / (1 + kl_penalty_factor)
#
#     psi_ref = kl_penalty_factor * (psi(ref_dist) - psi(ref_dist.sum())) / (1 + kl_penalty_factor)
#
#     for i in range(100):
#         for param_id in range(len(dist)):
#             v = psi(dist.sum()) + log_pk[param_id] + psi_ref[param_id]
#             dist[param_id] = inv_psi(v)
#
#     return dist
#
#
