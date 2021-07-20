
import numpy as np
from random import shuffle
import os
import errno
import datetime
import itertools
import glob
import datetime as dt
from shutil import copy
import csv
import time

#TODO
"""
state and action, filter out None
"""


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

random_normal_counter = 0
np_random_normal_cache = np.random.normal(size = (random_cache_size,))
def random_normal():
    global random_normal_counter, np_random_normal_cache, random_cache_size

    if random_normal_counter >= random_cache_size:
        random_normal_counter = 0
        np_random_normal_cache = np.random.normal(size = (random_cache_size,))

    val = np_random_normal_cache[random_normal_counter]
    random_normal_counter += 1
    return val



class ActionEnum:
    def __init__(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.STAY = 4

Action = ActionEnum()


class BasicLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate


    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, states, actions):
        return [self.learning_rate for _ in range(len(states))]

class ReducedLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, states, actions):
        n_steps =  len(states)
        return [self.learning_rate / n_steps for _ in range(len(states))]

class TrajMonteLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 100.):
        self.denoms = {key: 0. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon

    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_updates_elapsed = 0
        scheme.time_horizon = self.time_horizon

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]
        visitation = {}
        local_pressure = {}

        sum_of_sqr_rel_pressure = 0.
        sum_rel_pressure = 0.

        n_steps = len(states)

        for state, action in zip(states, actions):
            self.denoms[(state, action)] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(state, action)])
            )
            self.last_update_seen[(state, action)] = self.n_updates_elapsed

            visitation[(state, action)] = 0.
            local_pressure[(state, action)] = 0.


        for state, action in zip(states, actions):
            visitation[(state, action)] += 1. / n_steps


        for key in visitation:
            local_pressure[key] = (
                visitation[key]
                * visitation[key]
            )

            relative_pressure = (
                local_pressure[key]
                / (
                    local_pressure[key]
                    +  self.denoms[key]
                )
            )

            sum_of_sqr_rel_pressure += relative_pressure * relative_pressure
            sum_rel_pressure += relative_pressure

        step_size = sum_rel_pressure * sum_rel_pressure/ sum_of_sqr_rel_pressure


        for key in visitation:
            self.denoms[key] += step_size * local_pressure[key]


        for step_id, (state, action) in enumerate(zip(states, actions)):
            rates[step_id] = 1. / self.denoms[(state, action)]

        self.n_updates_elapsed += 1
        return rates

class SteppedMonteLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_updates_elapsed = 0
        scheme.time_horizon = self.time_horizon

        return scheme

    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]
        visited = []

        sum_of_sqr_rel_pressure = 0.
        sum_rel_pressure = 0.

        n_steps = len(states)

        for step_id, (state, action) in enumerate(zip(states, actions)):
            self.denoms[(state, action)][step_id] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(state, action)][step_id])
            )
            self.last_update_seen[(state, action)][step_id] = self.n_updates_elapsed

            visited.append(((state, action), step_id))

        for key, step_id in visited:
            relative_pressure = (1./ n_steps) / ((1./ n_steps) +  self.denoms[key][step_id])

            sum_of_sqr_rel_pressure += relative_pressure * relative_pressure
            sum_rel_pressure += relative_pressure

        step_size = (
            sum_rel_pressure * sum_rel_pressure
            / sum_of_sqr_rel_pressure
        )

        for key, step_id in visited:
            self.denoms[key][step_id] += step_size / (n_steps ** 2)


        for step_id, (state, action) in enumerate(zip(states, actions)):
            rates[step_id] = 1. / (self.denoms[(state, action)][step_id])

        self.n_updates_elapsed += 1
        return rates

class TrajTabularLearningRateScheme():
    def __init__(self, ref_model, has_only_state_as_key = False, time_horizon = 100.):
        self.denoms = {key: 0. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_state_as_key = has_only_state_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_state_as_key = self.has_only_state_as_key

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]

        for state, action in zip(states, actions):
            if self.has_only_state_as_key:
                self.denoms[state] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[state])
                )
                self.last_update_seen[state] = self.n_updates_elapsed

            else:
                self.denoms[(state, action)] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(state, action)])
                )
                self.last_update_seen[(state, action)] = self.n_updates_elapsed

        for state, action in zip(states, actions):
            if self.has_only_state_as_key:
                self.denoms[state] += 1

            else:
                self.denoms[(state, action)] += 1

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                rates[step_id] = 1. / self.denoms[state]

            else:
                rates[step_id] = 1. / self.denoms[(state, action)]

        self.n_updates_elapsed += 1
        return rates



class SteppedTabularLearningRateScheme():

    def __init__(self, ref_model, has_only_state_as_key = False, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_state_as_key = has_only_state_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_state_as_key = self.has_only_state_as_key

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                self.denoms[state][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[state][step_id])
                )
                self.last_update_seen[state][step_id] = self.n_updates_elapsed

            else:
                self.denoms[(state, action)][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(state, action)][step_id])
                )
                self.last_update_seen[(state, action)][step_id] = self.n_updates_elapsed

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                self.denoms[state][step_id] += 1

            else:
                self.denoms[(state, action)][step_id] += 1

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                rates[step_id] = 1. / self.denoms[state][step_id]

            else:
                rates[step_id] = 1. / self.denoms[(state, action)][step_id]

        self.n_updates_elapsed += 1
        return rates



class TrajCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = ReducedLearningRateScheme()
        self.core = {key: 0. for key in ref_model}


    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = self.core.copy()

        return critic

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[(state, action)]
        return evals

class SteppedCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[(state, action)]
            evals[step_id] = state_evals[step_id]
        return evals

class AveragedTrajCritic(TrajCritic):
    def eval(self, states, actions):
        return TrajCritic.eval(self, states, actions) / len(states)


class AveragedSteppedCritic(SteppedCritic):
    def eval(self, states, actions):
        return SteppedCritic.eval(self, states, actions) / len(states)


class MidTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        estimate = self.eval(states, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id] / n_steps
            self.core[(state, action)] += delta

class MidSteppedCritic(AveragedSteppedCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        estimate = self.eval(states, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id]  / n_steps
            self.core[(state, action)][step_id] += delta

class InexactMidTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.core[(state, action)] += delta


class InexactMidSteppedCritic(AveragedSteppedCritic):
    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.core[(state, action)][step_id] += delta

class QTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[(states[-1], actions[-1])] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[(states[step_id], actions[step_id])] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )


class QSteppedCritic(AveragedSteppedCritic):


    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[(states[-1], actions[-1])][-1] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[(states[step_id], actions[step_id])][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )

class BiQTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        if n_steps >= 2:
            self.core[(states[-1], actions[-1])] += (
                learning_rates[-1]
                * (
                    rewards[-1]
                    + 0.5 * step_evals[-2]
                    - step_evals[-1]
                )
            )

            self.core[(states[0], actions[0])] += (
                learning_rates[0]
                * (
                    rewards[0]
                    + 0.5 * step_evals[1]
                    - step_evals[0]
                )
            )


            for step_id in range(1, n_steps - 1):
                self.core[(states[step_id], actions[step_id])] += (
                    learning_rates[step_id]
                    * (
                        rewards[step_id]
                        + 0.5 * step_evals[step_id + 1]
                        + 0.5 * step_evals[step_id - 1]
                        - step_evals[step_id]
                    )
                )
        else:
            # nsteps = 1
            raise (
                NotImplementedError(
                    "BiQ is currently implemented for when the number of steps "
                    "is greater than 1."
                )
            )


class BiQSteppedCritic(AveragedSteppedCritic):


    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        if n_steps >= 2:
            self.core[(states[-1], actions[-1])][-1] += (
                learning_rates[-1]
                * (
                    rewards[-1]
                    + 0.5 * step_evals[-2]
                    - step_evals[-1]
                )
            )

            self.core[(states[0], actions[0])][0] += (
                learning_rates[0]
                * (
                    rewards[0]
                    + 0.5 * step_evals[1]
                    - step_evals[0]
                )
            )


            for step_id in range(1, n_steps - 1):
                self.core[(states[step_id], actions[step_id])][step_id] += (
                    learning_rates[step_id]
                    * (
                        rewards[step_id]
                        + 0.5 * step_evals[step_id + 1]
                        + 0.5 * step_evals[step_id - 1]
                        - step_evals[step_id]
                    )
                )
        else:
            # nsteps = 1
            raise (
                NotImplementedError(
                    "BiQ is currently implemented for when the number of steps "
                    "is greater than 1."
                )
            )

class VTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        self.core = {key: 0. for key in ref_model}
        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[state]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[-1]] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[states[step_id]] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )


class VSteppedCritic(AveragedSteppedCritic):

    def __init__(self, ref_model):
        self.core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.learning_rate_scheme = BasicLearningRateScheme()


    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic


    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[state]
            evals[step_id] = state_evals[step_id]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[-1]][-1] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[states[step_id]][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )

class UTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        self.core = {key: 0. for key in ref_model}

        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[state]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[0]] += (
            learning_rates[0]
            * (
                - step_evals[0]
            )
        )

        for step_id in range(1, n_steps):
            self.core[states[step_id]] += (
                learning_rates[step_id]
                * (
                    rewards[step_id - 1]
                    + step_evals[step_id - 1]
                    - step_evals[step_id]
                )
            )

class USteppedCritic(AveragedSteppedCritic):

    def __init__(self, ref_model):
        self.core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.learning_rate_scheme = BasicLearningRateScheme()

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic


    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[state]
            evals[step_id] = state_evals[step_id]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[0]][0] += (
            learning_rates[0]
            * (
                - step_evals[0]
            )
        )

        for step_id in range(1, n_steps):
            self.core[states[step_id]][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id - 1]
                    + step_evals[step_id - 1]
                    - step_evals[step_id]
                )

            )


class ABaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, states, actions, rewards):
        self.v_critic.update(states, actions, rewards)
        self.q_critic.update(states, actions, rewards)

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        q_step_evals = self.q_critic.step_evals(states, actions)
        v_step_evals = self.v_critic.step_evals(states, actions)
        return [q_step_evals[i] - v_step_evals[i] for i in range(len(q_step_evals))]

class ATrajCritic(ABaseCritic):
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QTrajCritic(ref_model_q)
        self.v_critic = VTrajCritic(ref_model_v)

    def copy(self):
        critic = self.__class__(self.q_critic.core, self.v_critic.core)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

class ASteppedCritic(ABaseCritic):
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QSteppedCritic(ref_model_q)
        self.v_critic = VSteppedCritic(ref_model_v)

    def copy(self):
        critic = self.__class__(self.q_critic.core, self.v_critic.core)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

class UqBaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, states, actions, rewards):
        self.u_critic.update(states, actions, rewards)
        self.q_critic.update(states, actions, rewards)

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions)) / len(states)

    def step_evals(self, states, actions):
        q_step_evals = self.q_critic.step_evals(states, actions)
        u_step_evals = self.u_critic.step_evals(states, actions)
        return [q_step_evals[i] + u_step_evals[i] for i in range(len(q_step_evals))]

class UqTrajCritic(UqBaseCritic):
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QTrajCritic(ref_model_q)
        self.u_critic = UTrajCritic(ref_model_u)

    def copy(self):
        critic = self.__class__(self.q_critic.core, self.u_critic.core)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic


class UqSteppedCritic(UqBaseCritic):
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QSteppedCritic(ref_model_q)
        self.u_critic = USteppedCritic(ref_model_u)

    def copy(self):
        critic = self.__class__(self.q_critic.core, self.u_critic.core)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

def target_cell_given_action(cell, action, n_rows, n_cols):
    row_id = cell[0]
    col_id = cell[1]

    if action == Action.STAY:
        return (row_id, col_id)

    if action == Action.LEFT:
        if col_id == 0:
            raise (
                ValueError(
                    f"Action LEFT is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return (row_id, col_id - 1)

    if action == Action.RIGHT:
        if col_id == n_cols - 1:
            raise (
                ValueError(
                    f"Action RIGHT is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return (row_id, col_id + 1)

    if action == Action.UP:
        if row_id == 0:
            raise (
                ValueError(
                    f"Action UP is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return  (row_id - 1, col_id)

    if action == Action.DOWN:
        if row_id == n_rows - 1:
            raise (
                ValueError(
                    f"Action DOWN is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return (row_id + 1, col_id)

def possible_actions_for_cell(cell, n_rows, n_cols):
    row_id = cell[0]
    col_id = cell[1]

    possible_actions = [Action.STAY]

    if col_id > 0:
        possible_actions.append(Action.LEFT)

    if col_id < n_cols - 1:
        possible_actions.append(Action.RIGHT)

    if row_id > 0:
        possible_actions.append(Action.UP)

    if row_id < n_rows - 1:
        possible_actions.append(Action.DOWN)

    return possible_actions

# def observation_cal(posns, agent_id):
#     other_agent_id = 1 - agent_id # 0 or 1 if agent is 1 or 0, respectfully.
#
#     observation = (
#         posns[agent_id][0],
#         posns[agent_id][1],
#         int(np.sign(posns[other_agent_id][0] - posns[agent_id][0])),
#         int(np.sign(posns[other_agent_id][1] - posns[agent_id][1])),
#     )
#
#     return observation

def observation_cal(posns, agent_id):
    other_agent_id = 1 - agent_id # 0 or 1 if agent is 1 or 0, respectfully.

    observation = (
        posns[agent_id][0],
        posns[agent_id][1],
        posns[other_agent_id][0],
        posns[other_agent_id][1],
    )

    return observation

def all_observations(n_rows, n_cols):
    for row_id in range(n_rows):
        for col_id in range(n_cols):
            for other_row_id in range(n_rows):
                for other_col_id in range(n_cols):
                    observation = (row_id, col_id, other_row_id, other_col_id)
                    yield observation

class Trajectory:
    def __init__(self, n_steps):
        self.rewards = [0. for i in range(n_steps)]
        self.observations = [None for i in range(n_steps)]
        self.actions = [None for i in range(n_steps)]

class Domain:
    def __init__(self, n_rows, n_cols, action_fail_rate, time_cost, reward_goal):
        self.n_steps = 50
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.action_fail_rate = action_fail_rate
        self.time_cost = time_cost # positive cost is negative reward
        self.reward_goal = reward_goal
        self.goal = (n_rows - 1, n_cols - 1)
        self.n_agents = 2


    def execute(self, policies):
        n_agents = self.n_agents
        n_steps = self.n_steps
        n_rows = self.n_rows
        n_cols = self.n_cols

        posns = [(0,0) for i in range(n_agents)]
        trajectories = [Trajectory(n_steps) for i in range(n_agents)]
        rewards = [0. for i in range(self.n_steps)]

        for step_id in range(n_steps):
            actions = [None for i in range(n_agents)]
            reward = -self.time_cost
            ending_early = False

            for agent_id in range(n_agents):
                observation = observation_cal(posns, agent_id)

                trajectories[agent_id].observations[step_id] = observation

                possible_actions = (
                    possible_actions_for_cell(posns[agent_id], n_rows, n_cols)
                )

                observation = observation_cal(posns, agent_id)
                action = policies[agent_id].action(observation)
                trajectories[agent_id].actions[step_id] = action

                if random_uniform() < self.action_fail_rate:
                    resulting_action = get_random_from_list(possible_actions)
                else:
                    resulting_action = action

                posns[agent_id] = (
                    target_cell_given_action(posns[agent_id], resulting_action, n_rows, n_cols)
                )


            if all(pos == self.goal for pos in posns):
                reward += self.reward_goal
                ending_early = True


            rewards[step_id] = reward

            if ending_early:
                break


        for agent_id in range(n_agents):
            trajectories[agent_id].rewards = rewards.copy()
            trajectories[agent_id].observations = list(filter(lambda x: x is not None, trajectories[agent_id].observations))
            trajectories[agent_id].actions = list(filter(lambda x: x is not None, trajectories[agent_id].actions))

        return trajectories


def phenotypes_from_population(population):
    phenotypes = [None] * len(population)

    for i in range(len(population)):
        phenotypes[i] = {"policy" : population[i]}

    return phenotypes

def population_from_phenotypes(phenotypes):
    population = [None] * len(phenotypes)

    for i in range(len(phenotypes)):
        population[i] = phenotypes[i]["policy"]

    return population


class Policy():
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.action_probabilities = {}

        for observation in all_observations(n_rows, n_cols):
            cell = (observation[0], observation[1])
            possible_actions = possible_actions_for_cell(cell, n_rows, n_cols)
            action_probs = [1. / len(possible_actions) for action_id in range(len(possible_actions))]
            self.action_probabilities[observation] = action_probs.copy()

    def copy(self):
        policy = Policy(self.n_rows, self.n_cols)

        policy.action_probabilities = self.action_probabilities.copy()

        return policy


    def action(self, observation):
        r = random_uniform()
        p = 0.
        cell = (observation[0], observation[1])
        possible_actions = possible_actions_for_cell(cell, self.n_rows, self.n_cols)
        action_probs = self.action_probabilities[observation]
        selected_action = possible_actions[0]

        for action_id, action in enumerate(possible_actions):
            selected_action = action
            p += action_probs[action_id]
            if p > r:
                break

        return selected_action

    def mutate(self, dist):
        for observation in all_observations(self.n_rows, self.n_cols):
            self.action_probabilities[observation] = np.random.dirichlet(dist[observation])

def create_dist(n_rows, n_cols, precision):
    dist = {}

    for observation in all_observations(n_rows, n_cols):
        cell = (observation[0], observation[1])
        possible_actions = possible_actions_for_cell(cell, n_rows, n_cols)
        dist[observation] = [precision for action_id in range(len(possible_actions))]

    return dist



def update_dist(dist, speed, sustain, phenotypes, n_rows, n_cols):

    phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])

    for i in range(len(phenotypes) // 2):
        policy = phenotypes[i]["policy"]
        trajectory = phenotypes[i]["trajectory"]

        observations = trajectory.observations
        actions = trajectory.actions

        for observation, action in zip(observations, actions):
            cell = (observation[0], observation[1])
            possible_actions = possible_actions_for_cell(cell, n_rows, n_cols)
            dist_observation = dist[observation]

            if len(dist_observation) != len(possible_actions):
                # Something went wrong
                raise RuntimeError("Something went wrong")

            for action_id in range(len(possible_actions)):
                if possible_actions[action_id] == action:
                    dist_observation[action_id] += speed

                dist_observation[action_id] *= sustain

    # for observation in dist.keys():
    #     dist_observation = dist[observation]
    #     for action_id in range(len(dist_observation)):
    #         dist_observation[action_id] += bonus_mark
