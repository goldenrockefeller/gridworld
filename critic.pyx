# cython: profile=False


# From the GECCO 2022 Paper on Bidirectional Fitness Critics:
#    The efficient ensemble method (EEFCLT)
#    is a combination of the Kalman learning rate scheme,
#    and the Combined Ensemble Critic


# TERMINOLOGY
# _______________
# See GECCO 2022 Paper for more information
# All "Critics" are Fitness Critics (although they *could* work as regular critics)
# All critics are functionally modelled as lookup-tables, (not neural networks).
# Tw is trajectory-wise updates
# Sw is step-wise updates
# Q is Q function
# V is Value Function
# U is the Reverse Value Function
# A is the Advantage Function (Q - V)
# Bi is the Bidirectional Value Function (Q + U)
# "targets" are step_wise targets, so Trajectory-wise update code must reflect this.
# "traj" means that the value is with respect to the all the time-steps (aka trajectory)
# "stepped" means that one value is with respect to one time-step
# "core" referes to the step-wise critic value mapping
# sometimes variance is refered to as "uncertainty" because that term is easier to understand
# unlike in the GECCO 2022 paper, "weights" are used as the inverse of variance (1/v), because that is easier to understand
# a "key" is either an observation, or an observation-action pair.


from typing import  Sequence, Iterable, Any, Optional, Hashable

import numpy as np
from libc.math cimport log, exp

# Cache random numbers for speed, and avoid python overhead.
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

def validate_trajectory_size(
        observations: Sequence[Hashable],
        actions: Sequence[Hashable],
        rewards: Optional[Sequence[Hashable]] = None,
):
    if len(observations) != len(actions):
        raise (
            ValueError(
                f"The number of observations (len(observations) = {len(observations)}) must "
                f"be equal to the number of actions (len(actions) = {len(actions)})."
            )
        )

    if rewards is not None:
        if len(observations) != len(rewards):
            raise (
                ValueError(
                    f"The number of observations (len(observations) = {len(observations)}) must "
                    f"be equal to the number of rewards (len(rewards) = {len(rewards)})."
                )
            )


def validate_trajectory_size_to_n_steps(
        n_steps,
        observations: Sequence[Hashable],
        actions: Sequence[Hashable],
        rewards: Optional[Sequence[Hashable]] = None,
):
    if len(observations) != n_steps:
        raise (
            ValueError(
                f"The number of observations (len(observations) = {len(observations)}) must "
                f"be equal to the number of steps (n_steps = {n_steps})."
            )
        )

    if len(actions) != n_steps:
        raise (
            ValueError(
                f"The number of actions (len(actions) = {len(actions)}) must "
                f"be equal to the number of steps (n_steps = {n_steps})."
            )
        )

    if rewards is not None:
        if len(rewards) != n_steps:
            raise (
                ValueError(
                    f"The number of rewards (len(rewards) = {len(rewards)}) must "
                    f"be equal to the number of steps (n_steps = {n_steps})."
                )
            )


def eligibility_trace_targets(rewards: Sequence[float], values: Sequence[float], trace_sustain: float) -> Sequence[float]:

    if len(rewards) != len(values):
        raise (
            ValueError(
                f"The number of rewards (len(rewards) = {len(rewards)}) must "
                f"be equal to the number of values (len(values) = {len(values)})."
            )
        )

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


def reverse_eligibility_trace_targets(rewards: Sequence[float], values: Sequence[float], trace_sustain: float) -> Sequence[float]:
    if len(rewards) != len(values):
        raise (
            ValueError(
                f"The number of rewards (len(rewards) = {len(rewards)}) must "
                f"be equal to the number of values (len(values) = {len(values)})."
            )
        )

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

class BasicLearningRateScheme():
    """Applies a constant learning rate"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate


    def copy(self):

        scheme = self.__class__(self.learning_rate)

        return scheme

    def learning_rates(self, observations: Sequence[Any], actions: Sequence[Any]) -> Sequence[float]:
        validate_trajectory_size(observations, actions)

        return [self.learning_rate for _ in range(len(observations))]

class ReducedLearningRateScheme():
    """Applies a the basic learning rate, scaled by the number of steps"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def copy(self):
        scheme = self.__class__(self.learning_rate)

        return scheme

    def learning_rates(self, observations: Sequence[Any], actions: Sequence[Any]) -> Sequence[float]:
        validate_trajectory_size(observations, actions)


        if len(observations) != len(actions):
            raise (
                ValueError(
                    f"The number of observations (len(observations) = {len(observations)}) must "
                    f"be equal to the number of actions (len(actions) = {len(actions)})."
                )
            )

        n_steps =  len(observations)
        return [self.learning_rate / n_steps for _ in range(len(observations))]

class TrajKalmanLearningRateScheme():
    """ Kalman filtering for paramater estimation using single value as target ('z') for all steps

    Each parameter is independent from each other.
    See wikipedia for explanation on what 'k', 'p', 'z' and 'h' are.
    The learning rate is the Kalman gain for the step.
    """

    def __init__(self, all_keys: Iterable[Hashable], has_only_observation_as_key = False):
        self.p = {key: float("inf") for key in all_keys}
        self.last_update_seen = {key: 0 for key in all_keys}
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

    def uncertainties(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:

        """ Returns parameters uncertanties"""
        validate_trajectory_size(observations, actions)

        uncertainties = [0. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            uncertainties[step_id] = self.p[key]

        return uncertainties

    def learning_rates(
            self,
            observations: Sequence[Hashable],
            actions: Sequence[Hashable],
            target_uncertainties: Optional[Sequence[float]] = None
    ) -> Sequence[float]:

        validate_trajectory_size(observations, actions)

        rates = [0. for _ in range(len(observations))]
        local_k = {}

        if target_uncertainties is None:
            target_uncertainties = [1. for _ in range(len(observations))]
        else:
            if len(observations) != len(target_uncertainties):
                raise (
                    ValueError(
                        f"The number of observations (len(observations) = {len(observations)}) must "
                        f"be equal to the number of target target_uncertainties "
                        f"(len(target_uncertainties) = {len(target_uncertainties)})."
                    )
                )


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
                k = p / (p + target_uncertainties[step_id])
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
    """ Kalman filtering for paramater estimation using single value as target ('z') for all steps


    Each parameter is not independent. They are used in a mean function h(x) targets z.
    See wikipedia for explanation on what 'k', 'p', 'z' and 'h' are.
    The learning rate is the Kalman gain for the step.
    """
    def __init__(self, all_keys: Iterable[Hashable], has_only_observation_as_key = False):
        self.p = {key: 1. for key in all_keys}
        self.last_update_seen = {key: 0 for key in all_keys}
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

    def uncertainties(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        """ Returns parameters uncertanties"""
        validate_trajectory_size(observations, actions)

        uncertainties = [0. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            uncertainties[step_id] = self.p[key]

        return uncertainties

    def learning_rates(
            self,
            observations: Sequence[Hashable],
            actions: Sequence[Hashable]
    ) -> Sequence[float]:
        validate_trajectory_size(observations, actions)

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
                n_inf_p += 1
                raise RuntimeError() # This shouldn't happen


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
    """ Kalman filtering for paramater estimation using multiple target values ('z').
     There is one target value per step


    Each parameter is independent.
    See wikipedia for explanation on what 'k', 'p', 'z' and 'h' are.
    The learning rate is the Kalman gain for the step.
    """

    def __init__(self, all_keys: Iterable[Hashable], n_steps, has_only_observation_as_key = False):
        self.p = {key: [float("inf") for _ in range(n_steps)] for key in all_keys}
        self.last_update_seen =  {key: [0 for _ in range(n_steps)] for key in all_keys}
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

    def uncertainties(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        """ Returns parameters uncertanties"""

        if len(observations) != len(actions):
            raise (
                ValueError(
                    f"The number of observations (len(observations) = {len(observations)}) must "
                    f"be equal to the number of actions (len(actions) = {len(actions)})."
                )
            )

        uncertainties = [0. for _ in range(len(observations))]

        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            uncertainties[step_id] = self.p[key][step_id]

        return uncertainties

    def learning_rates(
            self,
            observations: Sequence[Hashable],
            actions: Sequence[Hashable],
            target_uncertainties: Optional[Sequence[float]] = None
    ) -> Sequence[float]:
        if len(observations) != len(actions):
            raise (
                ValueError(
                    f"The number of observations (len(observations) = {len(observations)}) must "
                    f"be equal to the number of actions (len(actions) = {len(actions)})."
                )
            )

        rates = [0. for _ in range(len(observations))]

        if target_uncertainties is None:
            target_uncertainties = [1. for _ in range(len(observations))]
        else:
            if len(observations) != len(target_uncertainties):
                raise (
                    ValueError(
                        f"The number of observations (len(observations) = {len(observations)}) must "
                        f"be equal to the number of target target_uncertainties "
                        f"(len(target_uncertainties) = {len(target_uncertainties)})."
                    )
                )


        for step_id, observation, action in zip(range(len(observations)), observations, actions):
            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            p = self.p[key][step_id]

            if p == float("inf"):
                k = 1.
                p = target_uncertainties[step_id]
            else:
                p = (
                    p
                    + self.process_noise
                    * (self.n_process_steps_elapsed - self.last_update_seen[key][step_id])
                )
                k = p / (p + target_uncertainties[step_id])
                p = (1-k) * p

            self.p[key][step_id] = p

            self.last_update_seen[key][step_id] = self.n_process_steps_elapsed

            rates[step_id] = k

        return rates

    def advance_process(self):
        self.n_process_steps_elapsed += 1


def seq_mean(step_evals: Sequence[float]) -> float:
    return sum(step_evals) / len(step_evals)


class Critic:
    """Fitness Critic using lookup table as underlying functional model"""
    def __init__(self, all_keys: Iterable[Hashable], has_only_observation_as_key = False):
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core = {key: 0. for key in all_keys}
        self.has_only_observation_as_key = has_only_observation_as_key
        self.fn_aggregation = seq_mean # Note that aggregation  is a reference.

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = self.core.copy()
        critic.fn_aggregation = self.fn_aggregation # Note that aggregation is a reference.

        return critic

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError("Abstract Method")

    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size(observations, actions, rewards)
        n_steps = len(observations)
        targets = self.targets(observations, actions, rewards)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)



        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]

            if self.has_only_observation_as_key:
                key = observation
            else:
                key = (observation, action)

            delta = targets[step_id] - self.core[key]
            self.core[key] += learning_rates[step_id] * delta

    # def update(self, observations, actions, rewards):
    #     raise NotImplementedError("Abstract Method")

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:
        validate_trajectory_size(observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))

    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:

        validate_trajectory_size(observations, actions)

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

class TracedCritic(Critic):
    """
    Fitness Critic for eligibility traces using lookup table as underlying functional model
    """

    def __init__(self, all_keys: Iterable[Hashable], has_only_observation_as_key = False):
        Critic.__init__(self, all_keys, has_only_observation_as_key)
        self.trace_sustain = 0.

    def copy(self):
        critic = Critic.copy(self)
        critic.trace_sustain = self.trace_sustain

        return critic

cdef class EnsembleInfo:
    """All the Ensemble critic information that is associated with one observation-action pair.

    Compiled with Cython for speedy access/modification.
    Inherit at your own risk; there are no getters or setters to override.
    """
    cdef public double traj_value
    cdef public double traj_weight
    cdef public double traj_mul
    cdef public double traj_last_visited
    cdef public list stepped_values
    cdef public list stepped_weights
    cdef public list stepped_muls
    cdef public Py_ssize_t n_steps

    def __init__(self, n_steps):
        self.traj_value = 0.
        self.traj_weight = 0.
        self.traj_mul = 0.
        self.traj_last_visited = 0
        self.stepped_values = [0. for _ in range(n_steps)]
        self.stepped_weights = [0. for _ in range(n_steps)]
        self.stepped_muls = [0. for _ in range(n_steps)]
        self.n_steps = n_steps


    def copy(self):
        info = self.__class__(self.n_steps)

        info.traj_value = self.traj_value
        info.traj_weight = self.traj_weight
        info.traj_mul = self.traj_mul
        info.traj_last_visited = self.traj_last_visited
        info.stepped_values = self.stepped_values.copy()
        info.stepped_weights = self.stepped_weights.copy()
        info.stepped_muls = self.stepped_muls.copy()
        info.n_steps = self.n_steps

        return info

cdef class BaseEnsembleCritic():
    """The Ensemble Fitness Critic using lookup table as underlying functional model.
    This class is a template for both the regular Ensemble Critic and the Combined Ensemble Critic.

    Compiled with Cython for speedy access/modification.
    Inherit at your own risk; there are no getters or setters to override for some attributes.
    """
    cdef public object stepped_critic
    cdef public dict info
    cdef public double process_noise
    cdef public double n_process_steps_elapsed
    cdef public tuple init_params
    cdef public bint has_only_observation_as_key
    cdef public object aggregation
    cdef public Py_ssize_t n_steps

    def __init__(self, all_keys: Iterable[Hashable], n_steps, has_only_observation_as_key = False):
        self.info = {key: EnsembleInfo(n_steps) for key in all_keys}
        self.process_noise = 0.
        self.n_process_steps_elapsed = 0
        self.has_only_observation_as_key = has_only_observation_as_key
        self.fn_aggregation = seq_mean # Note that aggregation  is a reference.
        self.n_steps = n_steps


    def copy(self):
        critic = self.__class__(self.info, self.n_steps)
        critic.info = {key: self.info[key].copy() for key in self.info}
        critic.process_noise = self.process_noise
        critic.n_process_steps_elapsed = self.n_process_steps_elapsed
        critic.has_only_observation_as_key = self.has_only_observation_as_key
        critic.fn_aggregation = self.fn_aggregation # Note that aggregation is a reference.
        critic.n_steps = self.n_steps

        return critic


    def advance_process(self):
        self.n_process_steps_elapsed += 1

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:

        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))


    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)

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

    def stepped_values(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        cdef EnsembleInfo info

        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)

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


    def stepped_weights(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)

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


cdef class EnsembleCritic(BaseEnsembleCritic):
    """The Ensemble Fitness Critic using lookup table as underlying functional model.

    Compiled with Cython for speedy access/modification.
    Inherit at your own risk; there are no getters or setters to override for some attributes.
    """


    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError("Abstract Method")


    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):

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
        cdef EnsembleInfo info

        cdef list stepped_values
        cdef list stepped_weights
        cdef list stepped_muls



        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)


        targets = self.targets(observations, actions, rewards)

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
            target_weight = 1.

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


cdef class TracedEnsembleCritic(EnsembleCritic):
    """Ensemble Fitness Critic for eligibility traces using lookup table as underlying functional model

    Inherit at your own risk; there are no getters or setters to override.
    """
    cdef public double trace_sustain

    def __init__(self, all_keys: Iterable[Hashable], n_steps, has_only_observation_as_key = False):
        EnsembleCritic.__init__(self, all_keys, n_steps, has_only_observation_as_key)
        self.trace_sustain = 0.

    def copy(self):
        critic = EnsembleCritic.copy(self)
        critic.trace_sustain = self.trace_sustain

        return critic


cdef class CombinedEnsembleCritic(BaseEnsembleCritic):
    """The Combined Ensemble Fitness Critic using lookup table as underlying functional model.
        Instead of rewards, Combined Ensemble Fitness Critic uses the combined
        replacement value that results from multiple sub-step critic
        (e.g. For Advantage and Bidirectional Critics)

    Compiled with Cython for speedy access/modification.
    Inherit at your own risk; there are no getters or setters to override for some attributes.
    """

    def update(
            self,
            observations: Sequence[Hashable],
            actions: Sequence[Hashable],
            replacement_values: Sequence[float],
            replacement_weights: Sequence[float]
    ):
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
        cdef EnsembleInfo info

        cdef list stepped_values
        cdef list stepped_weights
        cdef list stepped_muls

        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)

        if len(replacement_weights) != self.n_steps:
            raise (
                ValueError(
                    f"The number of replacement weights (len(replacement_weights) = {len(replacement_weights)}) must "
                    f"be equal to the number of steps (self.n_steps = {self.n_steps})."
                )
            )

        if len(replacement_values) != self.n_steps:
            raise (
                ValueError(
                    f"The number of replacement values (len(replacement_values) = {len(replacement_values)}) must "
                    f"be equal to the number of steps (self.n_steps = {self.n_steps})."
                )
            )

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

cdef class TracedCombinedEnsembleCritic(CombinedEnsembleCritic):
    """Combined Ensemble Fitness Critic for eligibility traces using lookup table as underlying functional model

    Inherit at your own risk; there are no getters or setters to override.
    """
    cdef public double trace_sustain

    def __init__(self, all_keys: Iterable[Hashable], n_steps, has_only_observation_as_key = False):
        CombinedEnsembleCritic.__init__(self, all_keys, n_steps, has_only_observation_as_key)
        self.trace_sustain = 0.

    def copy(self):
        critic = CombinedEnsembleCritic.copy(self)
        critic.trace_sustain = self.trace_sustain

        return critic


class TwCritic(Critic):

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size(observations, actions, rewards)

        if self.fn_aggregation is not seq_mean:
            raise  (
                RuntimeError(
                    "The trajectory-wise fitness critic update is only defined for the mean aggregation function."
                    "(self.fn_aggregation is not seq_mean)"
                )
            )

        n_steps = len(observations)
        fitness = sum(rewards)
        estimate = self.eval(observations, actions)
        error = fitness - estimate
        step_evals = self.step_evals(observations, actions)

        # Note that these parameter targets (not fitness evalutation targets)
        targets = [step_evals[i] + error/n_steps for i in range(n_steps)]
        return targets

    # def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
    #     validate_trajectory_size(observations, actions, rewards)
    #
    #     n_steps = len(observations)
    #
    #     fitness = sum(rewards)
    #
    #     estimate = self.eval(observations, actions)
    #
    #     error = fitness - estimate
    #
    #     learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
    #
    #     for step_id in range(n_steps):
    #         observation = observations[step_id]
    #         action = actions[step_id]
    #         delta = error * learning_rates[step_id] / n_steps
    #         self.core[(observation, action)] += delta

class SwCritic(Critic):
    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size(observations, actions, rewards)
        n_steps = len(observations)
        fitness = sum(rewards)

        targets = [fitness for i in range(n_steps)]
        return targets

    # def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
    #     validate_trajectory_size(observations, actions, rewards)
    #     n_steps = len(observations)
    #
    #     fitness = sum(rewards)
    #
    #     step_evals = self.step_evals(observations, actions)
    #
    #     learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
    #
    #     for step_id in range(n_steps):
    #         observation = observations[step_id]
    #         action = actions[step_id]
    #         estimate = step_evals[step_id]
    #         error = fitness - estimate
    #         delta = error * learning_rates[step_id]
    #         self.core[(observation, action)] += delta

class SwEnsembleCritic(EnsembleCritic):
    """Inherit at your own risk; there are no getters or setters to override for some attributes."""
    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)

        sum_rewards = sum(rewards)
        targets = [sum_rewards for i in range(len(observations))]
        return targets

class QCritic(TracedCritic):
    def __init__(self, all_keys: Iterable[Hashable]):
        TracedCritic.__init__(self, all_keys)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(all_keys, False)

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size(observations, actions, rewards)
        n_steps = len(observations)

        values = self.step_evals(observations, actions)
        targets = eligibility_trace_targets(rewards, values, self.trace_sustain)
        return targets

    # def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
    #     validate_trajectory_size(observations, actions, rewards)
    #     n_steps = len(observations)
    #
    #     values = self.step_evals(observations, actions)
    #
    #     targets = eligibility_trace_targets(rewards, values, self.trace_sustain)
    #
    #     learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
    #
    #     for step_id in range(n_steps):
    #         delta = targets[step_id] - self.core[(observations[step_id], actions[step_id])]
    #         self.core[(observations[step_id], actions[step_id])] += learning_rates[step_id] * delta


class QEnsembleCritic(TracedEnsembleCritic):
    """Inherit at your own risk; there are no getters or setters to override for some attributes."""
    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)
        n_steps = len(observations)
        stepped_values = self.stepped_values(observations, actions)
        targets = eligibility_trace_targets(rewards, stepped_values, self.trace_sustain)

        return targets


class VCritic(TracedCritic):
    def __init__(self, all_keys: Iterable[Hashable]):
        TracedCritic.__init__(self, all_keys)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(all_keys, True)
        self.has_only_observation_as_key = True

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size(observations, actions, rewards)
        n_steps = len(observations)

        values = self.step_evals(observations, actions)
        targets = eligibility_trace_targets(rewards, values, self.trace_sustain)
        return targets

    # def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
    #     validate_trajectory_size(observations, actions, rewards)
    #     n_steps = len(observations)
    #
    #     values = self.step_evals(observations, actions)
    #
    #     targets = eligibility_trace_targets(rewards, values, self.trace_sustain)
    #
    #     learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
    #
    #
    #     for step_id in range(n_steps):
    #         delta = targets[step_id] - self.core[observations[step_id]]
    #         self.core[observations[step_id]] += learning_rates[step_id] * delta


class VEnsembleCritic(TracedEnsembleCritic):
    """Inherit at your own risk; there are no getters or setters to override for some attributes."""
    def __init__(self, all_keys: Iterable[Hashable], n_steps):
        TracedEnsembleCritic.__init__(self, all_keys, n_steps)
        self.has_only_observation_as_key = True

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)
        n_steps = len(observations)

        stepped_values = self.stepped_values(observations, actions)
        targets = eligibility_trace_targets(rewards, stepped_values, self.trace_sustain)

        return targets

class UCritic(TracedCritic):
    def __init__(self, all_keys):
        TracedCritic.__init__(self, all_keys)
        self.learning_rate_scheme = TrajKalmanLearningRateScheme(all_keys, True)
        self.has_only_observation_as_key = True

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size(observations, actions, rewards)
        n_steps = len(observations)

        values = self.step_evals(observations, actions)
        targets = reverse_eligibility_trace_targets(rewards, values, self.trace_sustain)
        return targets

    # def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
    #     validate_trajectory_size(observations, actions, rewards)
    #     n_steps = len(observations)
    #     values = self.step_evals(observations, actions)
    #
    #     targets = eligibility_trace_targets(rewards, values, self.trace_sustain)
    #
    #     learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)
    #
    #     for step_id in range(n_steps):
    #         delta = targets[step_id] - self.core[observations[step_id]]
    #         self.core[observations[step_id]] += learning_rates[step_id] * delta

class UEnsembleCritic(TracedEnsembleCritic):
    """Inherit at your own risk; there are no getters or setters to override for some attributes."""
    def __init__(self, all_keys: Iterable[Hashable], n_steps):
        TracedEnsembleCritic.__init__(self, all_keys, n_steps)
        self.has_only_observation_as_key = True

    def targets(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)
        n_steps = len(observations)
        stepped_values = self.stepped_values(observations, actions)
        targets = reverse_eligibility_trace_targets(rewards, stepped_values, self.trace_sustain)

        return targets

class ABaseCritic:
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def copy(self):
        raise NotImplementedError("Abstract Method")
        # critic = self.__class__(all_keys, all_keys)
        # critic.v_critic = self.v_critic.copy()
        # critic.q_critic = self.q_critic.copy()
        # critic.fn_aggregation = self.fn_aggregation
        #
        # return critic

    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size(observations, actions, rewards)
        self.v_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:
        validate_trajectory_size(observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))

    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size(observations, actions)
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

class ACritic(ABaseCritic):
    def __init__(self, all_q_keys: Iterable[Hashable], all_v_keys: Iterable[Hashable]):
        self.q_critic = QCritic(all_q_keys)
        self.v_critic = VCritic(all_v_keys)
        self.fn_aggregation = seq_mean
        self.all_q_keys = all_q_keys
        self.all_v_keys = all_v_keys

    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_v_keys)
        critic.q_critic = self.q_critic.copy()
        critic.v_critic = self.v_critic.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.all_q_keys = self.all_q_keys
        critic.all_v_keys = self.all_v_keys

class AEnsembleCritic(ABaseCritic):
    def __init__(self, all_q_keys: Iterable[Hashable], all_v_keys: Iterable[Hashable], n_steps):
        self.q_critic = QEnsembleCritic(all_q_keys, n_steps)
        self.v_critic = VEnsembleCritic(all_v_keys, n_steps)
        self.fn_aggregation = seq_mean
        self.n_steps = n_steps
        self.all_q_keys = all_q_keys
        self.all_v_keys = all_v_keys

    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_v_keys, self.n_steps)
        critic.q_critic = self.q_critic.copy()
        critic.v_critic = self.v_critic.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.n_steps = self.n_steps
        critic.all_q_keys = self.all_q_keys
        critic.all_v_keys = self.all_v_keys


class ACombinedEnsembleCritic:
    def __init__(self, all_q_keys: Iterable[Hashable],  all_v_keys: Iterable[Hashable], n_steps):
        self.q_critic = QEnsembleCritic(all_q_keys, n_steps)
        self.v_critic = VEnsembleCritic(all_v_keys, n_steps)
        self.core = CombinedEnsembleCritic(all_q_keys, n_steps)
        self.fn_aggregation = seq_mean
        self.n_steps = n_steps
        self.all_q_keys = all_q_keys
        self.all_v_keys = all_v_keys

    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_v_keys, self.n_steps)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()
        critic.core = self.core.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.n_steps = self.n_steps
        critic.all_q_keys = self.all_q_keys
        critic.all_v_keys = self.all_v_keys

        return critic

    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)
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

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))

    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)
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


class BiBaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def copy(self):
        raise NotImplementedError("Abstract Method")
        # critic = self.__class__(self.q_critic.all_keys, self.u_critic.all_keys)
        # critic.u_critic = self.u_critic.copy()
        # critic.q_critic = self.q_critic.copy()
        #
        # return critic

    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size(observations, actions, rewards)
        self.u_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:
        validate_trajectory_size(observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))

    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size(observations, actions)
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

class BiCritic(BiBaseCritic):
    def __init__(self, all_q_keys: Iterable[Hashable], all_u_keys: Iterable[Hashable]):
        self.q_critic = QCritic(all_q_keys)
        self.u_critic = UCritic(all_u_keys)
        self.fn_aggregation = seq_mean
        self.all_q_keys = all_q_keys
        self.all_u_keys = all_u_keys

    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_u_keys)
        critic.q_critic = self.q_critic.copy()
        critic.u_critic = self.u_critic.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.all_q_keys = self.all_q_keys
        critic.all_u_keys = self.all_u_keys



class BiEnsembleCritic(BiBaseCritic):
    def __init__(self, all_q_keys: Iterable[Hashable], all_u_keys: Iterable[Hashable], n_steps):
        self.q_critic = QEnsembleCritic(all_q_keys, n_steps)
        self.v_critic = VEnsembleCritic(all_u_keys, n_steps)
        self.fn_aggregation = seq_mean
        self.n_steps = n_steps
        self.all_q_keys = all_q_keys
        self.all_u_keys = all_u_keys

    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_u_keys, self.n_steps)
        critic.q_critic = self.q_critic.copy()
        critic.v_critic = self.v_critic.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.n_steps = self.n_steps
        critic.all_q_keys = self.all_q_keys
        critic.all_u_keys = self.all_u_keys

    # def __init__(self, all_keys_q, all_keys_u):
    #     self.q_critic = QEnsembleCritic(all_keys_q)
    #     self.u_critic = UEnsembleCritic(all_keys_u)

class BiCombinedEnsembleCritic:
    def __init__(self, all_q_keys: Iterable[Hashable], all_u_keys: Iterable[Hashable], n_steps):
        self.q_critic = QEnsembleCritic(all_q_keys, n_steps)
        self.u_critic = UEnsembleCritic(all_u_keys, n_steps)
        self.core = CombinedEnsembleCritic(all_q_keys, n_steps)
        self.fn_aggregation = seq_mean
        self.n_steps = n_steps
        self.all_q_keys = all_q_keys
        self.all_u_keys = all_u_keys


    def copy(self):
        critic = self.__class__(self.all_q_keys, self.all_u_keys, self.n_steps)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()
        critic.core = self.core.copy()
        critic.fn_aggregation = self.fn_aggregation
        critic.n_steps = self.n_steps
        critic.all_q_keys = self.all_q_keys
        critic.all_u_keys = self.all_u_keys

        return critic

    def update(self, observations: Sequence[Hashable], actions: Sequence[Hashable], rewards: Sequence[float]):
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions, rewards)
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

    def eval(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> float:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)
        return self.fn_aggregation(self.step_evals(observations, actions))

    def step_evals(self, observations: Sequence[Hashable], actions: Sequence[Hashable]) -> Sequence[float]:
        validate_trajectory_size_to_n_steps(self.n_steps, observations, actions)
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


#########################################################

    # def __init__(self, all_keys_q, all_keys_u):
    #     self.q_critic = QEnsembleCritic(all_keys_q)
    #     self.u_critic = UEnsembleCritic(all_keys_u)
    #     self.core = CombinedEnsembleCritic(all_keys_q)
    #
    # def copy(self):
    #     critic = self.__class__(self.q_critic.all_keys, self.u_critic.all_keys)
    #     critic.u_critic = self.u_critic.copy()
    #     critic.q_critic = self.q_critic.copy()
    #     critic.core = self.core.copy()
    #
    #     return critic
    #
    # def update(self, observations, actions, rewards):
    #     self.u_critic.update(observations, actions, rewards)
    #     self.q_critic.update(observations, actions, rewards)
    #
    #     u_values = self.u_critic.stepped_values(observations, actions)
    #     u_weights = self.u_critic.stepped_weights(observations, actions)
    #     q_values = self.q_critic.stepped_values(observations, actions)
    #     q_weights = self.q_critic.stepped_weights(observations, actions)
    #
    #     replacement_values = [0.] * len(u_values)
    #     replacement_weights = [0.] * len(u_weights)
    #
    #     for i in range(len(u_values)):
    #         replacement_values[i] = q_values[i] + u_values[i]
    #
    #         if q_weights[i] == 0. or u_weights[i] == 0.:
    #             replacement_weights[i] = 0.
    #         else:
    #             replacement_weights[i] = 1./ (1./q_weights[i] + 1./ u_weights[i])
    #
    #     self.core.update(observations, actions, replacement_values, replacement_weights)
    #
    # def eval(self, observations, actions):
    #     return sum(self.step_evals(observations, actions)) / len(observations)
    #
    # def step_evals(self, observations, actions):
    #     return self.core.step_evals(observations, actions)
    #
    # def advance_process(self):
    #     self.u_critic.advance_process()
    #     self.q_critic.advance_process()
    #     self.core.advance_process()
    #
    #
    # @property
    # def trace_sustain(self):
    #     raise RuntimeError()
    #
    # @trace_sustain.setter
    # def trace_sustain(self, val):
    #     a = self.u_critic.trace_sustain
    #     b = self.q_critic.trace_sustain
    #     self.u_critic.trace_sustain = val
    #     self.q_critic.trace_sustain = val
