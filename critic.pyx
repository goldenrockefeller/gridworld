# cython: profile=True

import numpy as np
from libc.math cimport log, exp

random_cache_size = 10000
random_uniform_counter = 0
np_random_uniform_cache = np.random.random(random_cache_size)
def random_uniform():
    global random_uniform_counter, np_random_uniform_cache, random_cache_size

    if random_uniform_counter >= random_cache_size:
        random_uniform_counter = 0
        np_random_uniform_cache = np.random.random(random_cache_size)

    val = np_random_uniform_cache[random_uniform_counter]
    random_uniform_counter += 1
    return val


def list_sum(my_list):
    val = 0.

    for d in my_list:
        val += d

    return val

def list_multiply(my_list, val):
    new_list = my_list.copy()

    for id, d in enumerate(my_list):
        my_list[id] = d * val

    return my_list


def apply_eligibility_trace(rewards, values, trace_sustain):
    targets = [0. for i in range(len(rewards))]
    n_steps = len(rewards)
    last_step_id = n_steps - 1

    for step_id in reversed(range(n_steps)):
        if step_id == last_step_id:
            targets[last_step_id] = rewards[last_step_id]
        else:
            targets[step_id] = (
                (1 - trace_sustain) *(rewards[step_id] + values[step_id + 1])
                + trace_sustain * (rewards[step_id] + targets[step_id+1])
            )

    return targets


def apply_reverse_eligibility_trace(rewards, values, trace_sustain):
    targets = [0. for i in range(len(rewards))]

    for step_id in range(len(rewards)):
        if step_id == 0:
            targets[step_id] = 0.
        else:
            targets[step_id] = (
                (1 - trace_sustain) *(rewards[step_id - 1] + values[step_id - 1])
                + trace_sustain * (rewards[step_id - 1] + targets[step_id-1])
            )

    return targets


# cdef fast_weighted_average(double v0, double u0, double v1, double u1):
#     cdef double total = 0.
#     cdef double total_weight = 0.
#     cdef double total_uncertainty = 0.
#     cdef double ratio_0 = 1. / u0
#
#     if u0 != float("inf"):
#         weight = 1. / u0
#
#         total += v0 * weight
#         total_weight += weight
#         total_uncertainty += u0 * weight * weight
#
#     if u1 != float("inf"):
#         weight = 1. / u1
#
#         total += v1 * weight
#         total_weight += weight
#         total_uncertainty += u1 * weight * weight
#
#
#
#     ratio_0 = ratio_0 / total_weight
#     total /= total_weight
#     total_uncertainty /= total_weight * total_weight
#
#     return total, total_uncertainty, ratio_0
#
#
# def weighted_average(value_uncertainties):
#     total = 0.
#     total_weight = 0.
#     total_uncertainty = 0.
#     ratio_0 = 1. / value_uncertainties[0][1]
#
#
#     for value, uncertainty in value_uncertainties:
#         if uncertainty != float("inf"):
#             weight = 1. / uncertainty
#
#             total += value * weight
#             total_weight += weight
#             total_uncertainty += uncertainty * weight * weight
#
#
#
#     ratio_0 = ratio_0 / total_weight
#     total /= total_weight
#     total_uncertainty /= total_weight * total_weight
#
#     return total, total_uncertainty, ratio_0
#
# def get_r_err(n_steps, cov):
#     vY = 0.
#     err = 0.
#     mul = 0.
#     for i in range(n_steps):
#         vY += 1 + 2 * mul * cov
#         mul += 1
#
#     return 1./ vY
#
#
#
# def apply_smart_eligibility_trace(rewards, values, uncertainties, cov = 0.):
#     trace = 0.
#     trace_uncertainty = 0.
#     n_steps = len(rewards)
#     last_step_id = len(rewards) - 1
#     targets = [ 0. for i in range(n_steps)]
#     targets_uncertainties = [ 0. for i in range(n_steps)]
#     r_err = get_r_err(n_steps, cov)
#     for step_id in reversed(range(n_steps)):
#         if step_id == last_step_id:
#             trace = rewards[step_id]
#             trace_uncertainty = r_err
#             ratio = 1.
#             mul = 1.
#         else:
#             pretarget = rewards[step_id] + values[step_id + 1]
#             pretarget_uncertainty = 1 / n_steps + uncertainties[step_id + 1]
#             trace += rewards[step_id]
#             trace_uncertainty += r_err + 2 * cov * mul * r_err
#             trace, trace_uncertainty, ratio = weighted_average(((trace, trace_uncertainty), (pretarget,  pretarget_uncertainty)))
#             mul *= ratio
#             mul += 1
#         targets[step_id] = trace
#         targets_uncertainties[step_id] = trace_uncertainty
#
#     return targets, targets_uncertainties
#
# def apply_smart_reverse_eligibility_trace(rewards, values, uncertainties, cov = 0.):
#     trace = 0.
#     trace_uncertainty = 0.
#     n_steps = len(rewards)
#     targets = [ 0. for i in range(n_steps)]
#     targets_uncertainties = [ 0. for i in range(n_steps)]
#     r_err = get_r_err(n_steps, cov)
#     for step_id in range(n_steps):
#         if step_id == 0.:
#             trace = 0.
#             trace_uncertainty = 0.
#             mul = 0.
#         else:
#             pretarget = rewards[step_id - 1] + values[step_id - 1]
#             pretarget_uncertainty = r_err + uncertainties[step_id - 1]
#
#             trace += rewards[step_id - 1]
#             trace_uncertainty += r_err + 2 * cov * mul * r_err
#             trace, targets_uncertainty, ratio = weighted_average(((trace, trace_uncertainty), (pretarget, pretarget_uncertainty)))
#             mul *= ratio
#             mul += 1
#         targets[step_id] = trace
#         if step_id == 0.:
#             targets_uncertainties[step_id] = 1 / n_steps
#         else:
#             targets_uncertainties[step_id] = trace_uncertainty
#     return targets, targets_uncertainties


# def apply_eligibility_trace(deltas, trace_sustain):
#     trace = 0.
#     for step_id in reversed(range(len(deltas))):
#         trace += deltas[step_id]
#         deltas[step_id]  = trace
#         trace *= trace_sustain
#
# def apply_reverse_eligibility_trace(deltas, trace_sustain):
#     trace = 0.
#     for step_id in range(len(deltas)):
#         trace += deltas[step_id]
#         deltas[step_id]  = trace
#         trace *= trace_sustain


# def make_light(heavy, observation_actions_x_episodes):
#     light = {}
#
#     for observation_actions in observation_actions_x_episodes:
#         for observation_action in observation_actions:
#             if observation_action not in light:
#                 light[observation_action] = {}
#
#         for step_id, observation_action in enumerate(observation_actions):
#             if step_id not in light[observation_action]:
#                 light[observation_action][step_id] = heavy[observation_action][step_id]
#
#     return light
#
# def make_light_o(heavy, observation_actions_x_episodes):
#     light = {}
#
#     for observation_actions in observation_actions_x_episodes:
#         for observation, action in observation_actions:
#             if observation not in light:
#                 light[observation] = {}
#
#         for step_id, observation_action in enumerate(observation_actions):
#             observation = observation_action[0]
#             if step_id not in light[observation]:
#                 light[observation][step_id] = heavy[observation][step_id]
#
#     return light
#
# def update_heavy(heavy_model, light_model):
#     new_model = {}
#
#     for key in light_model:
#         for step_id in light_model[key]:
#             heavy_model[key][step_id] = light_model[key][step_id]

class BasicLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate


    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, observations, actions):
        return [self.learning_rate for _ in range(len(observations))]

class ReducedLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, observations, actions):
        n_steps =  len(observations)
        return [self.learning_rate / n_steps for _ in range(len(observations))]
#
# class TrajMonteLearningRateScheme():
#
#     def __init__(self, ref_model, time_horizon = 100.):
#         self.denoms = {key: 0. for key in ref_model}
#         self.last_update_seen = {key: 0 for key in ref_model}
#         self.n_process_steps_elapsed = 0
#         self.time_horizon = time_horizon
#
#     def copy(self):
#         scheme = self.__class__(self.denoms)
#         scheme.denoms = self.denoms.copy()
#         scheme.last_update_seen = self.last_update_seen.copy()
#         scheme.n_process_steps_elapsed = 0
#         scheme.time_horizon = self.time_horizon
#
#         return scheme
#
#
#     def learning_rates(self, observations, actions):
#         rates = [0. for _ in range(len(observations))]
#         visitation = {}
#         local_pressure = {}
#
#
#         n_steps = len(observations)
#
#         for observation, action in zip(observations, actions):
#             self.denoms[(observation, action)] *= (
#                 (1. - 1. / self.time_horizon)
#                 ** (self.n_process_steps_elapsed - self.last_update_seen[(observation, action)])
#             )
#             self.last_update_seen[(observation, action)] = self.n_process_steps_elapsed
#
#             visitation[(observation, action)] = 0.
#             local_pressure[(observation, action)] = 0.
#
#
#         for observation, action in zip(observations, actions):
#             visitation[(observation, action)] += 1. / n_steps
#
#
#         for key in visitation:
#             local_pressure[key] = (
#                 visitation[key]
#                 * visitation[key]
#             )
#
#         for key in visitation:
#             self.denoms[key] += local_pressure[key]
#
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             rates[step_id] = 1. / self.denoms[(observation, action)]
#
#
#         return rates
#
# class SteppedMonteLearningRateScheme():
#
#     def __init__(self, ref_model, time_horizon = 100.):
#         self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.n_process_steps_elapsed = 0
#         self.time_horizon = time_horizon
#
#
#     def copy(self):
#         scheme = self.__class__(self.denoms)
#         scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
#         scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
#         scheme.n_process_steps_elapsed = 0
#         scheme.time_horizon = self.time_horizon
#
#         return scheme
#
#     def learning_rates(self, observations, actions):
#         rates = [0. for _ in range(len(observations))]
#         visited = []
#
#         n_steps = len(observations)
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             self.denoms[(observation, action)][step_id] *= (
#                 (1. - 1. / self.time_horizon)
#                 ** (self.n_process_steps_elapsed - self.last_update_seen[(observation, action)][step_id])
#             )
#             self.last_update_seen[(observation, action)][step_id] = self.n_process_steps_elapsed
#
#             visited.append(((observation, action), step_id))
#
#
#         for key, step_id in visited:
#             self.denoms[key][step_id] += 1. / (n_steps ** 2)
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             rates[step_id] = 1. / (self.denoms[(observation, action)][step_id])
#
#
#         return rates
#
#
#
# class TrajTabularLearningRateScheme():
#     def __init__(self, ref_model, has_only_observation_as_key = False, time_horizon = 100.):
#         self.denoms = {key: 0. for key in ref_model}
#         self.last_update_seen = {key: 0 for key in ref_model}
#         self.n_process_steps_elapsed = 0
#         self.time_horizon = time_horizon
#         self.has_only_observation_as_key = has_only_observation_as_key
#
#
#     def copy(self):
#         scheme = self.__class__(self.denoms)
#         scheme.denoms = self.denoms.copy()
#         scheme.last_update_seen = self.last_update_seen.copy()
#         scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
#         scheme.time_horizon = self.time_horizon
#         scheme.has_only_observation_as_key = self.has_only_observation_as_key
#
#         return scheme
#
#
#     def learning_rates(self, observations, actions):
#         rates = [0. for _ in range(len(observations))]
#
#         for observation, action in zip(observations, actions):
#             if self.has_only_observation_as_key:
#                 self.denoms[observation] *= (
#                     (1. - 1. / self.time_horizon)
#                     ** (self.n_process_steps_elapsed - self.last_update_seen[observation])
#                 )
#                 self.last_update_seen[observation] = self.n_process_steps_elapsed
#
#             else:
#                 self.denoms[(observation, action)] *= (
#                     (1. - 1. / self.time_horizon)
#                     ** (self.n_process_steps_elapsed - self.last_update_seen[(observation, action)])
#                 )
#                 self.last_update_seen[(observation, action)] = self.n_process_steps_elapsed
#
#         for observation, action in zip(observations, actions):
#             if self.has_only_observation_as_key:
#                 self.denoms[observation] += 1
#
#             else:
#                 self.denoms[(observation, action)] += 1
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             if self.has_only_observation_as_key:
#                 rates[step_id] = 1. / self.denoms[observation]
#
#             else:
#                 rates[step_id] = 1. / self.denoms[(observation, action)]
#
#
#         return rates
#
#
#
# class SteppedTabularLearningRateScheme():
#
#     def __init__(self, ref_model, has_only_observation_as_key = False, time_horizon = 100.):
#         self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.n_process_steps_elapsed = 0
#         self.time_horizon = time_horizon
#         self.has_only_observation_as_key = has_only_observation_as_key
#
#
#     def copy(self):
#         scheme = self.__class__(self.denoms)
#         scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
#         scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
#         scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
#         scheme.time_horizon = self.time_horizon
#         scheme.has_only_observation_as_key = self.has_only_observation_as_key
#
#         return scheme
#
#     def learning_rates(self, observations, actions):
#         rates = [0. for _ in range(len(observations))]
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             if self.has_only_observation_as_key:
#                 self.denoms[observation][step_id] *= (
#                     (1. - 1. / self.time_horizon)
#                     ** (self.n_process_steps_elapsed - self.last_update_seen[observation][step_id])
#                 )
#                 self.last_update_seen[observation][step_id] = self.n_process_steps_elapsed
#
#             else:
#                 self.denoms[(observation, action)][step_id] *= (
#                     (1. - 1. / self.time_horizon)
#                     ** (self.n_process_steps_elapsed - self.last_update_seen[(observation, action)][step_id])
#                 )
#                 self.last_update_seen[(observation, action)][step_id] = self.n_process_steps_elapsed
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             if self.has_only_observation_as_key:
#                 self.denoms[observation][step_id] += 1
#
#             else:
#                 self.denoms[(observation, action)][step_id] += 1
#
#         for step_id, (observation, action) in enumerate(zip(observations, actions)):
#             if self.has_only_observation_as_key:
#                 rates[step_id] = 1. / self.denoms[observation][step_id]
#
#             else:
#                 rates[step_id] = 1. / self.denoms[(observation, action)][step_id]
#
#
#         return rates
#
#     # def learning_rates(self, list observations, list actions):
#     #     cdef list rates = [0. for _ in range(len(observations))]
#     #     cdef Py_ssize_t step_id
#     #     cdef double time_horizon =self.time_horizon
#     #     cdef double n_process_steps_elapsed = self.n_process_steps_elapsed
#     #     cdef dict denoms = self.denoms
#     #     cdef dict last_update_seen = self.last_update_seen
#     #     cdef bint has_only_observation_as_key = self.has_only_observation_as_key
#     #
#     #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
#     #         if has_only_observation_as_key:
#     #             denoms[observation][step_id] *= (
#     #                 (1. - 1. / time_horizon)
#     #                 ** (n_process_steps_elapsed - last_update_seen[observation][step_id])
#     #             )
#     #             last_update_seen[observation][step_id] = n_process_steps_elapsed
#     #
#     #         else:
#     #             denoms[(observation, action)][step_id] *= (
#     #                 (1. - 1. / time_horizon)
#     #                 ** (n_process_steps_elapsed - last_update_seen[(observation, action)][step_id])
#     #             )
#     #             last_update_seen[(observation, action)][step_id] = n_process_steps_elapsed
#     #
#     #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
#     #         if has_only_observation_as_key:
#     #             denoms[observation][step_id] += 1
#     #
#     #         else:
#     #             denoms[(observation, action)][step_id] += 1
#     #
#     #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
#     #         if has_only_observation_as_key:
#     #             rates[step_id] = 1. / denoms[observation][step_id]
#     #
#     #         else:
#     #             rates[step_id] = 1. / denoms[(observation, action)][step_id]
#     #
#     #     n_process_steps_elapsed += 1
#     #     return rates
#
#     def make_light(self, observation_actions_x_episodes):
#         light = self.__class__.__new__(self.__class__)
#
#         if self.has_only_observation_as_key:
#             light.denoms = make_light_o(self.denoms, observation_actions_x_episodes)
#             light.last_update_seen =  make_light_o(self.denoms, observation_actions_x_episodes)
#         else:
#             light.denoms = make_light(self.denoms, observation_actions_x_episodes)
#             light.last_update_seen =  make_light(self.denoms, observation_actions_x_episodes)
#         light.n_process_steps_elapsed = self.n_process_steps_elapsed
#         light.time_horizon = self.time_horizon
#         light.has_only_observation_as_key = self.has_only_observation_as_key
#
#         return light
#
#
#     def update_heavy(self, light):
#         update_heavy(self.denoms, light.denoms)
#         update_heavy(self.last_update_seen, light.last_update_seen)
#         self.n_process_steps_elapsed = light.n_process_steps_elapsed
#         self.time_horizon = light.time_horizon
#         self.has_only_observation_as_key = light.has_only_observation_as_key



class TrajKalmanLearningRateScheme():
    def __init__(self, ref_model, has_only_observation_as_key = False):
        self.p = {key: float("inf") for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_process_steps_elapsed = 0
        self.process_noise = 0.
        self.has_only_observation_as_key = has_only_observation_as_key


    def copy(self):
        scheme = self.__class__(self.p)
        scheme.p = self.p.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
        scheme.process_noise = self.process_noise
        scheme.has_only_observation_as_key = self.has_only_observation_as_key

        return scheme

    def uncertainties(self, observations, actions):
        uncertainties = [0. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            uncertainties[step_id] = self.p[key]

        return uncertainties

    def learning_rates(self, observations, actions, uncertainties = None):
        rates = [0. for _ in range(len(observations))]
        local_k = {}

        if uncertainties is None:
            uncertainties = [1. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            p = self.p[key]

            if p == float("inf"):
                k = 1.
                p = 1.
            else:
                p = (
                    p
                    + self.process_noise
                    * (self.n_process_steps_elapsed - self.last_update_seen[key])
                )
                k = p / (p + uncertainties[step_id])
                p = (1-k) * p

            self.p[key] = p

            self.last_update_seen[key] = self.n_process_steps_elapsed

            # set initial (prior) local_k
            local_k[key] = k

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            rates[step_id] = local_k[key]


        return rates

    def advance_process(self):
        self.n_process_steps_elapsed += 1

class MeanTrajKalmanLearningRateScheme():
    def __init__(self, ref_model, has_only_observation_as_key = False):
        self.p = {key: 1. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_process_steps_elapsed = 0
        self.process_noise = 0.
        self.has_only_observation_as_key = has_only_observation_as_key


    def copy(self):
        scheme = self.__class__(self.p)
        scheme.p = self.p.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
        scheme.process_noise = self.process_noise
        scheme.has_only_observation_as_key = self.has_only_observation_as_key

        return scheme


    def learning_rates(self, observations, actions):
        n_steps = len(observations)
        rates = [0. for _ in range(len(observations))]
        local_h = {}
        local_p = {}

        n_inf_p = 0

        for observation, action in zip(observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            local_h[key] = 0.

        for observation, action in zip(observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)


            local_h[key] += 1. / n_steps

            p = self.p[key]

            if p != float("inf"):
                p = (
                    p
                    + self.process_noise / n_steps
                    * (self.n_process_steps_elapsed - self.last_update_seen[key])
                )

            self.p[key] = p
            local_p[key] = p

            self.last_update_seen[key] = self.n_process_steps_elapsed

        for key in local_p:
            p = local_p[key]
            if p == float("inf"):
                raise ValueError()
                n_inf_p += 1

        if n_inf_p == 0:
            denom = 1.
            for key in local_p:
                denom += local_h[key] * local_h[key] * local_p[key]
        else:
            denom = 0.
            nom = 1.
            for key in local_p:
                p = local_p[key]
                if p == float("inf"):
                    denom += local_h[key] * local_h[key]
                else:
                    nom += local_h[key] * local_h[key] * local_p[key]


        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)


            if n_inf_p > 0:
                if local_p[key] == float("inf"):
                    p = nom / (local_h[key] * local_h[key])
                    rates[step_id] = 1. / (local_h[key] * local_h[key])
                else:
                    p = local_p[key]
                    rates[step_id] = 0.
            else:
                k = local_h[key] * local_p[key] / denom
                p = (1-k * local_h[key]) * p
                rates[step_id] = k / local_h[key]

            self.p[key] = p


        return rates

    def advance_process(self):
        self.n_process_steps_elapsed += 1


class SteppedKalmanLearningRateScheme():

    def __init__(self, ref_model, has_only_observation_as_key = False):
        self.p = {key: [float("inf") for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_process_steps_elapsed = 0
        self.process_noise = 0.
        self.has_only_observation_as_key = has_only_observation_as_key


    def copy(self):
        scheme = self.__class__(self.p)
        scheme.p = {key : self.p[key].copy() for key in self.p}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
        scheme.process_noise = self.process_noise
        scheme.has_only_observation_as_key = self.has_only_observation_as_key

        return scheme

    def uncertainties(self, observations, actions):
        uncertainties = [0. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            uncertainties[step_id] = self.p[key][step_id]

        return uncertainties

    def learning_rates(self, observations, actions, uncertainties = None):
        rates = [0. for _ in range(len(observations))]

        if uncertainties is None:
            uncertainties = [1. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            p = self.p[key][step_id]

            if p == float("inf"):
                k = 1.
                p = uncertainties[step_id]
            else:
                p = (
                    p
                    + self.process_noise
                    * (self.n_process_steps_elapsed - self.last_update_seen[key][step_id])
                )
                k = p / (p + uncertainties[step_id])
                p = (1-k) * p

            self.p[key][step_id] = p

            self.last_update_seen[key][step_id] = self.n_process_steps_elapsed

            rates[step_id] = k

        return rates

    def advance_process(self):
        self.n_process_steps_elapsed += 1

# class MeanSteppedKalmanLearningRateScheme():
#
#     def __init__(self, ref_model, has_only_observation_as_key = False):
#         self.p = {key: [float("inf") for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.n_process_steps_elapsed = 0
#         self.process_noise = 0.
#         self.has_only_observation_as_key = has_only_observation_as_key
#
#
#     def copy(self):
#         scheme = self.__class__(self.p)
#         scheme.p = {key : self.p[key].copy() for key in self.p}
#         scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
#         scheme.n_process_steps_elapsed = self.n_process_steps_elapsed
#         scheme.process_noise = self.process_noise
#         scheme.has_only_observation_as_key = self.has_only_observation_as_key
#
#         return scheme
#
#
#     def learning_rates(self, observations, actions):
#         n_steps = len(observations)
#
#         rates = [0. for _ in range(len(observations))]
#
#         local_p = {}
#
#         n_inf_p = 0
#         denom = 1.
#         h = 1. / n_steps
#
#         for step_id, observation, action in zip(range(len(observations)), observations, actions):
#             if self.has_only_observation_as_key:
#                 key = observation
#             else:
#                 key = (observation, action)
#
#             h = 1. / n_steps
#
#             p = self.p[key][step_id]
#
#             if p == float("inf"):
#                 n_inf_p += 1
#             else:
#                 p = (
#                     p
#                     + self.process_noise
#                     * (self.n_process_steps_elapsed - self.last_update_seen[key][step_id])
#                 )
#                 denom += h * h * p
#
#             self.p[key][step_id] = p
#
#             self.last_update_seen[key][step_id] = self.n_process_steps_elapsed
#
#
#
#         for step_id, observation, action in zip(range(len(observations)), observations, actions):
#             if self.has_only_observation_as_key:
#                 key = observation
#             else:
#                 key = (observation, action)
#
#             p = self.p[key][step_id]
#
#             if n_inf_p > 0:
#                 if local_p[key] == float("inf"):
#                     k = 1.
#                     p = 1. / (h * h)
#                 else:
#                     k = 0.
#                     p = self.p[key]
#             else:
#                 k = h * p / denom
#                 p = (1-k *h) * p
#
#
#             self.p[key] = p
#             rates[step_id] = k / h
#
#
#         return rates
#
#     def advance_process(self):
#         self.n_process_steps_elapsed += 1
#

class TrajCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = ReducedLearningRateScheme()
        self.core = {key: 0. for key in ref_model}
        self.has_only_observation_as_key = False
        self.trace_sustain = None
        self.ref_model = ref_model

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = self.core.copy()
        critic.trace_sustain = self.trace_sustain
        critic.ref_model = self.ref_model

        return critic

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]

            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            evals[step_id] = self.core[key]
        return evals

    def advance_process(self):
        self.learning_rate_scheme.advance_process()



    # @property
    # def time_horizon(self):
    #     return self.learning_rate_scheme.time_horizon
    #
    # @time_horizon.setter
    # def time_horizon(self, val):
    #     self.learning_rate_scheme.time_horizon = val

# class SteppedCritic():
#     def __init__(self, ref_model):
#         self.learning_rate_scheme = BasicLearningRateScheme()
#         self.core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.has_only_observation_as_key = False
#
#     def copy(self):
#         critic = self.__class__(self.core)
#         critic.learning_rate_scheme = self.learning_rate_scheme.copy()
#         critic.core = {key : self.core[key].copy() for key in self.core.keys()}
#
#         return critic
#
#     def eval(self, observations, actions):
#         return list_sum(self.step_evals(observations, actions))
#
#     def step_evals(self, observations, actions):
#         evals = [0. for _ in range(len(observations))]
#         for step_id in range(len(observations)):
#             observation = observations[step_id]
#             action = actions[step_id]
#
#             if self.has_only_observation_as_key:
#                 key = observation
#             else:
#                 key = (observation, action)
#
#             evals[step_id] = self.core[key][step_id]
#         return evals
#
#     def advance_process(self):
#         self.learning_rate_scheme.advance_process()
#
#     # @property
#     # def time_horizon(self):
#     #     return self.learning_rate_scheme.time_horizon
#     #
#     # @time_horizon.setter
#     # def time_horizon(self, val):
#     #     self.learning_rate_scheme.time_horizon = val

cdef class HybridInfo:
    cdef public double traj_value
    cdef public double traj_weight
    cdef public double traj_mul
    cdef public double traj_last_visited
    cdef public list stepped_values
    cdef public list stepped_weights
    cdef public list stepped_muls
    cdef public Py_ssize_t size

    def __init__(self, size):
        self.traj_value = 0.
        self.traj_weight = 0.
        self.traj_mul = 0.
        self.traj_last_visited = 0
        self.stepped_values = [0. for _ in range(size)]
        self.stepped_weights = [0. for _ in range(size)]
        self.stepped_muls = [0. for _ in range(size)]
        self.size = size


    def copy(self):
        info = self.__class__(self.size)

        info.traj_value = self.traj_value
        info.traj_weight = self.traj_weight
        info.traj_mul = self.traj_mul
        info.traj_last_visited = self.traj_last_visited
        info.stepped_values = self.stepped_values.copy()
        info.stepped_weights = self.stepped_weights.copy()
        info.stepped_muls = self.stepped_muls.copy()
        info.size = self.size

        return info

cdef class HybridCritic():
    cdef public object stepped_critic
    cdef public dict info
    cdef public double process_noise
    cdef public double n_process_steps_elapsed
    cdef public tuple init_params

    def __init__(self, ref_model):
        self.info = {key: HybridInfo(len(ref_model[key])) for key in ref_model}
        self.process_noise = 0.
        self.n_process_steps_elapsed = 0
        self.ref_model = ref_model
        self.trace_sustain = None
        self.has_only_observation_as_key = False


    def copy(self):
        critic = self.__class__(self.ref_model)
        critic.info = {key: self.info[key].copy() for key in self.info}
        critic.process_noise = self.process_noise
        critic.n_process_steps_elapsed = self.n_process_steps_elapsed
        critic.ref_model = self.ref_model
        critic.trace_sustain = self.trace_sustain
        critic.has_only_observation_as_key = self.has_only_observation_as_key

        return critic

    def update(self, observations, actions, rewards):
        cdef list targets
        cdef list target_uncertainties
        cdef Py_ssize_t step_id
        cdef object key
        cdef double target_value
        cdef double target_weight

        cdef double traj_value
        cdef double traj_weight
        cdef double traj_mul
        cdef double old_uncertainty
        cdef double new_uncertainty

        cdef double stepped_value
        cdef double stepped_weight
        cdef double stepped_mul
        cdef double last_stepped_value
        cdef double last_stepped_weight
        cdef HybridInfo info

        cdef list stepped_values
        cdef list stepped_weights
        cdef list stepped_muls


        targets, target_uncertainties = self.targets(observations, actions, rewards)

        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]

            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            info = self.info[key]
            stepped_values = info.stepped_values
            stepped_weights = info.stepped_weights
            stepped_muls = info.stepped_muls

            target_value = targets[step_id]
            target_weight = 1. / target_uncertainties[step_id]

            traj_weight = info.traj_weight
            traj_mul = info.traj_mul
            traj_value = info.traj_value

            if traj_weight > 0.:
                old_uncertainty = 1. / traj_weight
                new_uncertainty = old_uncertainty + self.process_noise * (self.n_process_steps_elapsed - info.traj_last_visited)
                traj_mul += log(old_uncertainty / new_uncertainty)
                traj_weight = 1. / new_uncertainty


            stepped_value = stepped_values[step_id]
            stepped_weight = stepped_weights[step_id]
            stepped_mul = stepped_muls[step_id]
            stepped_weight *= exp(traj_mul - stepped_mul)
            stepped_mul = traj_mul

            last_stepped_value = stepped_value
            last_stepped_weight = stepped_weight

            stepped_value = (stepped_value * stepped_weight + target_value * target_weight) / (stepped_weight + target_weight)
            stepped_weight = stepped_weight + target_weight

            traj_value = (
                (traj_value * traj_weight - last_stepped_value * last_stepped_weight + stepped_value * stepped_weight)
                / (traj_weight -  last_stepped_weight + stepped_weight)
            )
            traj_weight = traj_weight - last_stepped_weight + stepped_weight

            info.traj_mul = traj_mul
            info.traj_value = traj_value
            info.traj_weight = traj_weight
            stepped_values[step_id] = stepped_value
            stepped_weights[step_id] = stepped_weight
            stepped_muls[step_id] = stepped_mul

            info.traj_last_visited = self.n_process_steps_elapsed

    def advance_process(self):
        self.n_process_steps_elapsed += 1

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))


    def step_evals(self, list observations, list actions):

        evals = [0. for _ in range(len(observations))]

        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            evals[step_id] = self.info[key].traj_value

        return evals

    def stepped_values(self, observations, actions):
        cdef HybridInfo info

        values = [0. for _ in range(len(observations))]

        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            info = self.info[key]
            values[step_id] = info.stepped_values[step_id]

        return values


    def stepped_weights(self, observations, actions):
        weights = [0. for _ in range(len(observations))]

        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            weights[step_id] = self.info[key].stepped_weights[step_id]

        return weights



cdef class CombinedHybridCritic():
    cdef public object stepped_critic
    cdef public dict info
    cdef public double process_noise
    cdef public double n_process_steps_elapsed
    cdef public tuple init_params

    def __init__(self, ref_model):
        self.info = {key: HybridInfo(len(ref_model[key])) for key in ref_model}
        self.process_noise = 0.
        self.n_process_steps_elapsed = 0
        self.ref_model = ref_model
        self.trace_sustain = None
        self.has_only_observation_as_key = False


    def copy(self):
        critic = self.__class__(self.ref_model)
        critic.info = {key: self.info[key].copy() for key in self.info}
        critic.process_noise = self.process_noise
        critic.n_process_steps_elapsed = self.n_process_steps_elapsed
        critic.ref_model = self.ref_model
        critic.trace_sustain = self.trace_sustain
        critic.has_only_observation_as_key = self.has_only_observation_as_key

        return critic

    def update(self, observations, actions, replacement_values, replacement_weights):
        cdef list targets
        cdef list target_uncertainties
        cdef Py_ssize_t step_id
        cdef object key

        cdef double traj_value
        cdef double traj_weight
        cdef double traj_mul
        cdef double old_uncertainty
        cdef double new_uncertainty

        cdef double replacement_value
        cdef double replacement_weight

        cdef double stepped_value
        cdef double stepped_weight
        cdef double stepped_mul
        cdef double last_stepped_value
        cdef double last_stepped_weight
        cdef HybridInfo info

        cdef list stepped_values
        cdef list stepped_weights
        cdef list stepped_muls


        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]

            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            info = self.info[key]
            stepped_values = info.stepped_values
            stepped_weights = info.stepped_weights
            stepped_muls = info.stepped_muls

            replacement_value = replacement_values[step_id]
            replacement_weight = replacement_weights[step_id]

            traj_weight = info.traj_weight
            traj_mul = info.traj_mul
            traj_value = info.traj_value

            if traj_weight > 0.:
                old_uncertainty = 1. / traj_weight
                new_uncertainty = old_uncertainty + self.process_noise * (self.n_process_steps_elapsed - info.traj_last_visited)
                traj_mul += log(old_uncertainty / new_uncertainty)
                traj_weight = 1. / new_uncertainty


            stepped_value = stepped_values[step_id]
            stepped_weight = stepped_weights[step_id]
            stepped_mul = stepped_muls[step_id]
            stepped_weight *= exp(traj_mul - stepped_mul)
            stepped_mul = traj_mul

            last_stepped_value = stepped_value
            last_stepped_weight = stepped_weight

            stepped_value = replacement_value
            stepped_weight = replacement_weight

            traj_value = (
                (traj_value * traj_weight - last_stepped_value * last_stepped_weight + stepped_value * stepped_weight)
                / (traj_weight -  last_stepped_weight + stepped_weight)
            )
            traj_weight = traj_weight - last_stepped_weight + stepped_weight

            info.traj_mul = traj_mul
            info.traj_value = traj_value
            info.traj_weight = traj_weight
            stepped_values[step_id] = stepped_value
            stepped_weights[step_id] = stepped_weight
            stepped_muls[step_id] = stepped_mul

            info.traj_last_visited = self.n_process_steps_elapsed

    def advance_process(self):
        self.n_process_steps_elapsed += 1

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))


    def step_evals(self, list observations, list actions):

        evals = [0. for _ in range(len(observations))]

        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            evals[step_id] = self.info[key].traj_value

        return evals

class AveragedTrajCritic(TrajCritic):
    def eval(self, observations, actions):
        return TrajCritic.eval(self, observations, actions) / len(observations)

# class AveragedSteppedCritic(SteppedCritic):
#     def eval(self, observations, actions):
#         return SteppedCritic.eval(self, observations, actions) / len(observations)

class AveragedHybridCritic(HybridCritic):
    def eval(self, observations, actions):
        return HybridCritic.eval(self, observations, actions) / len(observations)

class AveragedCombinedHybridCritic(CombinedHybridCritic):
    def eval(self, observations, actions):
        return CombinedHybridCritic.eval(self, observations, actions) / len(observations)

class MidTrajCritic(AveragedTrajCritic):

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        fitness = list_sum(rewards)

        estimate = self.eval(observations, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id] / n_steps
            self.core[(observation, action)] += delta

# class MidSteppedCritic(AveragedSteppedCritic):
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         fitness = list_sum(rewards)
#
#         estimate = self.eval(observations, actions)
#
#         error = fitness - estimate
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         for step_id in range(n_steps):
#             observation = observations[step_id]
#             action = actions[step_id]
#             delta = error * learning_rates[step_id] / n_steps
#             self.core[(observation, action)][step_id] += delta
#
# class MidHybridCritic(AveragedHybridCritic):
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         fitness = list_sum(rewards)
#
#         estimate = list_sum(self.step_evals_from_stepped(observations, actions)) / n_steps
#
#         error = fitness - estimate
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         for step_id in range(n_steps):
#             observation = observations[step_id]
#             action = actions[step_id]
#             delta = error * learning_rates[step_id] / n_steps
#             self.stepped_core[(observation, action)][step_id] += delta
#             self.traj_core[(observation,action)]  += delta / len(self.stepped_core[(observation, action)])

class InexactMidTrajCritic(AveragedTrajCritic):

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        fitness = list_sum(rewards)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.core[(observation, action)] += delta


# class InexactMidSteppedCritic(AveragedSteppedCritic):
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         fitness = list_sum(rewards)
#
#         step_evals = self.step_evals(observations, actions)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         for step_id in range(n_steps):
#             observation = observations[step_id]
#             action = actions[step_id]
#             estimate = step_evals[step_id]
#             error = fitness - estimate
#             delta = error * learning_rates[step_id]
#             self.core[(observation, action)][step_id] += delta


class InexactMidHybridCritic(AveragedHybridCritic):
    def targets(self, observations, actions, rewards):

        sum_rewards = sum(rewards)
        targets = [sum_rewards for i in range(len(observations))]
        target_uncertainties = [1. for i in range(len(observations))]

        return targets, target_uncertainties

class QTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        AveragedTrajCritic.__init__(self, ref_model)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(ref_model, False)

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        values = self.step_evals(observations, actions)

        targets = apply_eligibility_trace(rewards, values, self.trace_sustain)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            delta = targets[step_id] - self.core[(observations[step_id], actions[step_id])]
            self.core[(observations[step_id], actions[step_id])] += learning_rates[step_id] * delta


# class QSteppedCritic(AveragedSteppedCritic):
#     def __init__(self, ref_model):
#         AveragedSteppedCritic.__init__(self, ref_model)
#         self.learning_rate_scheme = SteppedKalmanLearningRateScheme(ref_model, False)
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         uncertainties = self.learning_rate_scheme.uncertainties(observations, actions)
#         values = self.step_evals(observations, actions)
#
#         targets, target_uncertianties = apply_smart_eligibility_trace(rewards, values, uncertainties)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions, target_uncertianties)
#
#         for step_id in range(n_steps):
#             delta = targets[step_id] - self.core[(observations[step_id], actions[step_id])][step_id]
#             self.core[(observations[step_id], actions[step_id])][step_id] += learning_rates[step_id] * delta

class QHybridCritic(AveragedHybridCritic):
    def targets(self, observations, actions,rewards):
        n_steps = len(observations)
        stepped_values = self.stepped_values(observations, actions)
        targets = apply_eligibility_trace(rewards, stepped_values, self.trace_sustain)
        target_uncertainties = [1. for i in range(len(observations))]

        return targets, target_uncertainties

# class BiQTrajCritic(AveragedTrajCritic):
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         step_evals = self.step_evals(observations, actions)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         if n_steps >= 2:
#             self.core[(observations[-1], actions[-1])] += (
#                 learning_rates[-1]
#                 * (
#                     rewards[-1]
#                     + 0.5 * step_evals[-2]
#                     - step_evals[-1]
#                 )
#             )
#
#             self.core[(observations[0], actions[0])] += (
#                 learning_rates[0]
#                 * (
#                     rewards[0]
#                     + 0.5 * step_evals[1]
#                     - step_evals[0]
#                 )
#             )
#
#
#             for step_id in range(1, n_steps - 1):
#                 self.core[(observations[step_id], actions[step_id])] += (
#                     learning_rates[step_id]
#                     * (
#                         rewards[step_id]
#                         + 0.5 * step_evals[step_id + 1]
#                         + 0.5 * step_evals[step_id - 1]
#                         - step_evals[step_id]
#                     )
#                 )
#         else:
#             # nsteps = 1
#             raise (
#                 NotImplementedError(
#                     "BiQ is currently implemented for when the number of steps "
#                     "is greater than 1."
#                 )
#             )
#
#
# class BiQSteppedCritic(AveragedSteppedCritic):
#
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         step_evals = self.step_evals(observations, actions)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         if n_steps >= 2:
#             self.core[(observations[-1], actions[-1])][-1] += (
#                 learning_rates[-1]
#                 * (
#                     rewards[-1]
#                     + 0.5 * step_evals[-2]
#                     - step_evals[-1]
#                 )
#             )
#
#             self.core[(observations[0], actions[0])][0] += (
#                 learning_rates[0]
#                 * (
#                     rewards[0]
#                     + 0.5 * step_evals[1]
#                     - step_evals[0]
#                 )
#             )
#
#
#             for step_id in range(1, n_steps - 1):
#                 self.core[(observations[step_id], actions[step_id])][step_id] += (
#                     learning_rates[step_id]
#                     * (
#                         rewards[step_id]
#                         + 0.5 * step_evals[step_id + 1]
#                         + 0.5 * step_evals[step_id - 1]
#                         - step_evals[step_id]
#                     )
#                 )
#         else:
#             # nsteps = 1
#             raise (
#                 NotImplementedError(
#                     "BiQ is currently implemented for when the number of steps "
#                     "is greater than 1."
#                 )
#             )

class VTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        AveragedTrajCritic.__init__(self, ref_model)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(ref_model, True)
        self.has_only_observation_as_key = True

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        values = self.step_evals(observations, actions)

        targets = apply_eligibility_trace(rewards, values, self.trace_sustain)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)



        for step_id in range(n_steps):
            delta = targets[step_id] - self.core[observations[step_id]]
            self.core[observations[step_id]] += learning_rates[step_id] * delta


class VHybridCritic(AveragedHybridCritic):
    def __init__(self, ref_model):
        AveragedHybridCritic.__init__(self, ref_model)
        self.has_only_observation_as_key = True

    def targets(self, observations, actions,rewards):
        n_steps = len(observations)

        stepped_values = self.stepped_values(observations, actions)
        targets = apply_eligibility_trace(rewards, stepped_values, self.trace_sustain)
        target_uncertainties = [1. for i in range(len(observations))]

        return targets, target_uncertainties

# class VSteppedCritic(AveragedSteppedCritic):
#
#     def __init__(self, ref_model):
#         AveragedSteppedCritic.__init__(self, ref_model)
#         self.learning_rate_scheme = SteppedKalmanLearningRateScheme(ref_model, True)
#         self.has_only_observation_as_key = True
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         uncertainties = self.learning_rate_scheme.uncertainties(observations, actions)
#         values = self.step_evals(observations, actions)
#
#         targets, target_uncertianties = apply_smart_eligibility_trace(rewards, values, uncertainties)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions, target_uncertianties)
#
#         for step_id in range(n_steps):
#             delta = targets[step_id] - self.core[observations[step_id]][step_id]
#             self.core[observations[step_id]][step_id] += learning_rates[step_id] * delta


#
#
# class VHybridCritic(AveragedHybridCritic):
#
#     def __init__(self, ref_model):
#         self.traj_core = {key: 0. for key in ref_model}
#         self.stepped_core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.learning_rate_scheme = BasicLearningRateScheme()
#         self.trace_sustain = 0.
#
#
#     def copy(self):
#         critic = self.__class__(self.stepped_core)
#         critic.learning_rate_scheme = self.learning_rate_scheme.copy()
#         critic.stepped_core = {key : self.stepped_core[key].copy() for key in self.stepped_core.keys()}
#         critic.traj_core = self.traj_core.copy()
#         critic.trace_sustain = self.trace_sustain
#
#         return critic
#
#     def step_evals(self, observations, actions):
#         evals = [0. for _ in range(len(observations))]
#         for step_id in range(len(observations)):
#             observation = observations[step_id]
#             action = actions[step_id]
#             evals[step_id] = self.traj_core[observation]
#         return evals
#
#     def step_evals_from_stepped(self, observations, actions):
#         evals = [0. for _ in range(len(observations))]
#         for step_id in range(len(observations)):
#             observation = observations[step_id]
#             action = actions[step_id]
#             observation_evals = self.stepped_core[observation]
#             evals[step_id] = observation_evals[step_id]
#         return evals
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         step_evals = self.step_evals_from_stepped(observations, actions)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#         deltas = [0.] * n_steps
#
#         deltas[-1] =  rewards[-1] - step_evals[-1]
#
#         for step_id in range(n_steps - 1):
#             deltas[step_id] = (
#                 rewards[step_id]
#                 + step_evals[step_id + 1]
#                 - step_evals[step_id]
#             )
#
#         apply_eligibility_trace(deltas, self.trace_sustain)
#
#         self.stepped_core[observations[-1]][-1] += learning_rates[-1] * deltas[-1]
#         self.traj_core[observations[-1]] += learning_rates[-1] * deltas[-1] / len(self.stepped_core[observations[-1]])
#
#         for step_id in range(n_steps - 1):
#             self.stepped_core[observations[step_id]][step_id] +=  learning_rates[step_id] * deltas[step_id]
#             self.traj_core[observations[step_id]] +=  learning_rates[step_id] * deltas[step_id] / len(self.stepped_core[observations[step_id]])
#
#
#         # Fully reset Traj Core every so often to address quantization noise.
#         if random_uniform() < 0.1 / len(self.traj_core):
#             for observation in self.traj_core:
#                 total = 0.
#                 for stepped_val in self.stepped_core[observation]:
#                     total += stepped_val
#                 self.traj_core[observation] = total / len(self.stepped_core[observation])

class UTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        AveragedTrajCritic.__init__(self, ref_model)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(ref_model, True)
        self.has_only_observation_as_key = True

    def update(self, observations, actions, rewards):
        n_steps = len(observations)
        values = self.step_evals(observations, actions)

        targets = apply_eligibility_trace(rewards, values, self.trace_sustain)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            delta = targets[step_id] - self.core[observations[step_id]]
            self.core[observations[step_id]] += learning_rates[step_id] * delta

class UHybridCritic(AveragedHybridCritic):
    def __init__(self, ref_model):
        AveragedHybridCritic.__init__(self, ref_model)
        self.has_only_observation_as_key = True

    def targets(self, observations, actions,rewards):
        n_steps = len(observations)
        stepped_values = self.stepped_values(observations, actions)
        targets = apply_reverse_eligibility_trace(rewards, stepped_values, self.trace_sustain)
        target_uncertainties = [1. for i in range(len(observations))]

        return targets, target_uncertainties

# class USteppedCritic(AveragedSteppedCritic):
#
#     def __init__(self, ref_model):
#         AveragedSteppedCritic.__init__(self, ref_model)
#         self.learning_rate_scheme = SteppedKalmanLearningRateScheme(ref_model, True)
#         self.has_only_observation_as_key = True
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         uncertainties = self.learning_rate_scheme.uncertainties(observations, actions)
#         values = self.step_evals(observations, actions)
#
#         targets, target_uncertianties = apply_smart_reverse_eligibility_trace(rewards, values, uncertainties)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions, target_uncertianties)
#
#         for step_id in range(n_steps):
#             delta = targets[step_id] - self.core[observations[step_id]][step_id]
#             self.core[observations[step_id]][step_id] += learning_rates[step_id] * delta
#
# class UHybridCritic(AveragedHybridCritic):
#
#     def __init__(self, ref_model):
#         self.traj_core = {key: 0. for key in ref_model}
#         self.stepped_core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
#         self.learning_rate_scheme = BasicLearningRateScheme()
#         self.trace_sustain = 0.
#
#
#     def copy(self):
#         critic = self.__class__(self.stepped_core)
#         critic.learning_rate_scheme = self.learning_rate_scheme.copy()
#         critic.stepped_core = {key : self.stepped_core[key].copy() for key in self.stepped_core.keys()}
#         critic.traj_core = self.traj_core.copy()
#         critic.trace_sustain = self.trace_sustain
#
#         return critic
#
#
#
#
#     def step_evals(self, observations, actions):
#         evals = [0. for _ in range(len(observations))]
#         for step_id in range(len(observations)):
#             observation = observations[step_id]
#             action = actions[step_id]
#             evals[step_id] = self.traj_core[observation]
#         return evals
#
#     def step_evals_from_stepped(self, observations, actions):
#
#         evals = [0. for _ in range(len(observations))]
#         for step_id in range(len(observations)):
#             observation = observations[step_id]
#             action = actions[step_id]
#             observation_evals = self.stepped_core[observation]
#             evals[step_id] = observation_evals[step_id]
#         return evals
#
#
#     def update(self, observations, actions, rewards):
#         n_steps = len(observations)
#
#         step_evals = self.step_evals_from_stepped(observations, actions)
#
#         learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
#
#
#         deltas = [0.] * n_steps
#
#         deltas[0] = -step_evals[0]
#
#
#         for step_id in range(1, n_steps):
#             deltas[step_id] = (
#                 rewards[step_id - 1]
#                 + step_evals[step_id - 1]
#                 - step_evals[step_id]
#             )
#
#         apply_reverse_eligibility_trace(deltas, self.trace_sustain)
#
#
#         self.stepped_core[observations[0]][0] += learning_rates[0] * deltas[0]
#         self.traj_core[observations[0]] += learning_rates[0] * deltas[0] / len(self.stepped_core[observations[0]])
#
#         for step_id in range(1, n_steps):
#             self.stepped_core[observations[step_id]][step_id] += learning_rates[step_id] * deltas[step_id]
#             self.traj_core[observations[step_id]] += learning_rates[step_id] * deltas[step_id] / len(self.stepped_core[observations[step_id]])
#
#         # Fully reset Traj Core every so often to address quantization noise.
#         if random_uniform() < 0.1 / len(self.traj_core):
#             for observation in self.traj_core:
#                 total = 0.
#                 for stepped_val in self.stepped_core[observation]:
#                     total += stepped_val
#                 self.traj_core[observation] = total / len(self.stepped_core[observation])
#
#     def make_light(self, observation_actions_x_episodes):
#         light = self.__class__.__new__(self.__class__)
#
#         light.learning_rate_scheme = self.learning_rate_scheme.make_light(observation_actions_x_episodes)
#         light.stepped_core = make_light_o(self.stepped_core, observation_actions_x_episodes)
#         light.traj_core = self.traj_core.copy()
#
#         light.trace_sustain = self.trace_sustain
#
#         return light
#
#     def update_heavy(self, light):
#         AveragedHybridCritic.update_heavy(self, light)
#         self.trace_sustain = light.trace_sustain




class ABaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def copy(self):
        critic = self.__class__(self.q_critic.ref_model, self.v_critic.ref_model)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

    def update(self, observations, actions, rewards):
        self.v_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        q_step_evals = self.q_critic.step_evals(observations, actions)
        v_step_evals = self.v_critic.step_evals(observations, actions)
        return [q_step_evals[i] - v_step_evals[i] for i in range(len(q_step_evals))]

    def advance_process(self):
        self.v_critic.advance_process()
        self.q_critic.advance_process()


    @property
    def trace_sustain(self):
        raise RuntimeError()

    @trace_sustain.setter
    def trace_sustain(self, val):
        a = self.v_critic.trace_sustain
        b = self.q_critic.trace_sustain
        self.v_critic.trace_sustain = val
        self.q_critic.trace_sustain = val

    # @property
    # def time_horizon(self):
    #     raise NotImplementedError()
    #
    # @time_horizon.setter
    # def time_horizon(self, val):
    #     self.v_critic.learning_rate_scheme.time_horizon = val
    #     self.q_critic.learning_rate_scheme.time_horizon = val

class ATrajCritic(ABaseCritic):
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QTrajCritic(ref_model_q)
        self.v_critic = VTrajCritic(ref_model_v)



#
# class ASteppedCritic(ABaseCritic):
#     def __init__(self, ref_model_q, ref_model_v):
#         self.q_critic = QSteppedCritic(ref_model_q)
#         self.v_critic = VSteppedCritic(ref_model_v)



class AHybridCritic(ABaseCritic):
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QHybridCritic(ref_model_q)
        self.v_critic = VHybridCritic(ref_model_v)


class ACombinedHybridCritic:
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QHybridCritic(ref_model_q)
        self.v_critic = VHybridCritic(ref_model_v)
        self.core = AveragedCombinedHybridCritic(ref_model_q)

    def copy(self):
        critic = self.__class__(self.q_critic.ref_model, self.v_critic.ref_model)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()
        critic.core = self.core.copy()

        return critic

    def update(self, observations, actions, rewards):
        self.v_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

        v_values = self.v_critic.stepped_values(observations, actions)
        v_weights = self.v_critic.stepped_weights(observations, actions)
        q_values = self.q_critic.stepped_values(observations, actions)
        q_weights = self.q_critic.stepped_weights(observations, actions)

        replacement_values = [0.] * len(v_values)
        replacement_weights = [0.] * len(v_weights)

        for i in range(len(v_values)):
            replacement_values[i] = q_values[i] - v_values[i]

            if q_weights[i] == 0. or v_weights[i] == 0.:
                replacement_weights[i] = 0.
            else:
                replacement_weights[i] = 1./ (1./q_weights[i] + 1./ v_weights[i])

        self.core.update(observations, actions, replacement_values, replacement_weights)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions)) / len(observations)

    def step_evals(self, observations, actions):
        return self.core.step_evals(observations, actions)

    def advance_process(self):
        self.v_critic.advance_process()
        self.q_critic.advance_process()
        self.core.advance_process()


    @property
    def trace_sustain(self):
        raise RuntimeError()

    @trace_sustain.setter
    def trace_sustain(self, val):
        a = self.v_critic.trace_sustain
        b = self.q_critic.trace_sustain
        self.v_critic.trace_sustain = val
        self.q_critic.trace_sustain = val


class UqBaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def copy(self):
        critic = self.__class__(self.q_critic.ref_model, self.u_critic.ref_model)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic


    def update(self, observations, actions, rewards):
        self.u_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions)) / len(observations)

    def step_evals(self, observations, actions):
        q_step_evals = self.q_critic.step_evals(observations, actions)
        u_step_evals = self.u_critic.step_evals(observations, actions)
        return [q_step_evals[i] + u_step_evals[i] for i in range(len(q_step_evals))]

    def advance_process(self):
        self.u_critic.advance_process()
        self.q_critic.advance_process()


    @property
    def trace_sustain(self):
        raise RuntimeError()

    @trace_sustain.setter
    def trace_sustain(self, val):
        a = self.u_critic.trace_sustain
        b = self.q_critic.trace_sustain
        self.u_critic.trace_sustain = val
        self.q_critic.trace_sustain = val


    # @property
    # def time_horizon(self):
    #     raise NotImplementedError()
    #
    # @time_horizon.setter
    # def time_horizon(self, val):
    #     self.u_critic.learning_rate_scheme.time_horizon = val
    #     self.q_critic.learning_rate_scheme.time_horizon = val

class UqTrajCritic(UqBaseCritic):
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QTrajCritic(ref_model_q)
        self.u_critic = UTrajCritic(ref_model_u)
#
# class UqSteppedCritic(UqBaseCritic):
#     def __init__(self, ref_model_q, ref_model_u):
#         self.q_critic = QSteppedCritic(ref_model_q)
#         self.u_critic = USteppedCritic(ref_model_u)
#
#     def advance_process(self):
#         self.q_critic.advance_process()
#         self.u_critic.advance_process()



class UqHybridCritic(UqBaseCritic):
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QHybridCritic(ref_model_q)
        self.u_critic = UHybridCritic(ref_model_u)



class UqCombinedHybridCritic:
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QHybridCritic(ref_model_q)
        self.u_critic = UHybridCritic(ref_model_u)
        self.core = AveragedCombinedHybridCritic(ref_model_q)

    def copy(self):
        critic = self.__class__(self.q_critic.ref_model, self.u_critic.ref_model)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()
        critic.core = self.core.copy()

        return critic

    def update(self, observations, actions, rewards):
        self.u_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

        u_values = self.u_critic.stepped_values(observations, actions)
        u_weights = self.u_critic.stepped_weights(observations, actions)
        q_values = self.q_critic.stepped_values(observations, actions)
        q_weights = self.q_critic.stepped_weights(observations, actions)

        replacement_values = [0.] * len(u_values)
        replacement_weights = [0.] * len(u_weights)

        for i in range(len(u_values)):
            replacement_values[i] = q_values[i] + u_values[i]

            if q_weights[i] == 0. or u_weights[i] == 0.:
                replacement_weights[i] = 0.
            else:
                replacement_weights[i] = 1./ (1./q_weights[i] + 1./ u_weights[i])

        self.core.update(observations, actions, replacement_values, replacement_weights)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions)) / len(observations)

    def step_evals(self, observations, actions):
        return self.core.step_evals(observations, actions)

    def advance_process(self):
        self.u_critic.advance_process()
        self.q_critic.advance_process()
        self.core.advance_process()


    @property
    def trace_sustain(self):
        raise RuntimeError()

    @trace_sustain.setter
    def trace_sustain(self, val):
        a = self.u_critic.trace_sustain
        b = self.q_critic.trace_sustain
        self.u_critic.trace_sustain = val
        self.q_critic.trace_sustain = val



    # def copy(self):
    #     critic = self.__class__(self.q_critic.stepped_core, self.u_critic.stepped_core)
    #     critic.u_critic = self.u_critic.copy()
    #     critic.q_critic = self.q_critic.copy()
    #
    #     return critic
    #
    # def make_light(self, observation_actions_x_episodes):
    #
    #     light = self.__class__.__new__(self.__class__)
    #
    #     light.q_critic = self.q_critic.make_light(observation_actions_x_episodes)
    #     light.u_critic = self.u_critic.make_light(observation_actions_x_episodes)
    #
    #     return light
    #
    #
    # def update_heavy(self, light):
    #     self.q_critic.update_heavy(light.q_critic)
    #     self.u_critic.update_heavy(light.u_critic)


