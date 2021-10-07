from scipy.special import zeta, psi
from libc.math cimport isfinite, exp
import numpy as np
from numpy import log

cdef double inf = np.inf

def trigamma(x):
    # From https://en.wikipedia.org/wiki/Polygamma_function
    return zeta(2, x)

cdef double my_sum(double[:] arr):
    cdef double total = 0.

    for i in range(arr.shape[0]):
        total += arr[i]

    return total

def my_optimize(ref_dist, double kl_penalty_factor, list data):
    # Using fixed point method http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    cdef double[:] dist = ref_dist.copy()

    log_pk = 0. * data[0]

    for datum in data:
        log_pk += log(datum + 1e-9)
    log_pk /= len(data)

    cdef double[:] log_pk_c = log_pk / (1 + kl_penalty_factor)

    cdef double[:] psi_ref = kl_penalty_factor * (psi(ref_dist) - psi(ref_dist.sum())) / (1 + kl_penalty_factor)
    #

    cdef double v
    for i in range(20):
        for param_id in range(len(dist)):
            v = psi(my_sum(dist)) + (log_pk_c[param_id] + psi_ref[param_id])
            dist[param_id] = inv_psi(v)

    return np.asarray( dist)


cdef double inv_psi(double x):
    cdef double s = x
    cdef double f
    cdef double prev_s = -inf
    cdef double prev_s_mid = -inf

    cdef double s_mid = -inf
    cdef double s_hi = inf
    cdef double s_lo = -inf

    for i in range(1000):
        prev_s = s
        f = psi(exp(s)) - x
        s = s - (f / (trigamma( exp(s)) * exp(s)))



        f = psi(exp(s)) - x

        if isfinite(s):
            if (prev_s == s):
                return(exp(s))

            if f == 0:
                return exp(s)



        if f > 0:
            s_hi = min(s, s_hi)
            s = s_hi
        else:
            s_lo = max(s, s_lo)
            s = s_lo


        prev_s_mid = s_mid
        s_mid = 0.5 * (s_hi + s_lo)

        if isfinite(s_mid) :
            f = psi(exp(s_mid)) - x

            if f == 0:
                return exp(s_mid)

            if f > 0:
                s_hi = min(s_mid, s_hi)
                s_mid = s_hi
            else:
                s_lo = max(s_mid, s_lo)
                s_mid = s_lo

            if (prev_s_mid == s_mid):
                return(exp(s_mid))


    return exp(s)
