# cython: profile=True

import numpy as np

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


def apply_eligibility_trace(deltas, trace_sustain):
    trace = 0.
    for step_id in reversed(range(len(deltas))):
        trace += deltas[step_id]
        deltas[step_id]  = trace
        trace *= trace_sustain

def apply_reverse_eligibility_trace(deltas, trace_sustain):
    trace = 0.
    for step_id in range(len(deltas)):
        trace += deltas[step_id]
        deltas[step_id]  = trace
        trace *= trace_sustain


def make_light(heavy, observation_actions_x_episodes):
    light = {}

    for observation_actions in observation_actions_x_episodes:
        for observation_action in observation_actions:
            if observation_action not in light:
                light[observation_action] = {}

        for step_id, observation_action in enumerate(observation_actions):
            if step_id not in light[observation_action]:
                light[observation_action][step_id] = heavy[observation_action][step_id]

    return light

def make_light_o(heavy, observation_actions_x_episodes):
    light = {}

    for observation_actions in observation_actions_x_episodes:
        for observation, action in observation_actions:
            if observation not in light:
                light[observation] = {}

        for step_id, observation_action in enumerate(observation_actions):
            observation = observation_action[0]
            if step_id not in light[observation]:
                light[observation][step_id] = heavy[observation][step_id]

    return light

def update_heavy(heavy_model, light_model):
    new_model = {}

    for key in light_model:
        for step_id in light_model[key]:
            heavy_model[key][step_id] = light_model[key][step_id]

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


    def learning_rates(self, observations, actions):
        rates = [0. for _ in range(len(observations))]
        visitation = {}
        local_pressure = {}


        n_steps = len(observations)

        for observation, action in zip(observations, actions):
            self.denoms[(observation, action)] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(observation, action)])
            )
            self.last_update_seen[(observation, action)] = self.n_updates_elapsed

            visitation[(observation, action)] = 0.
            local_pressure[(observation, action)] = 0.


        for observation, action in zip(observations, actions):
            visitation[(observation, action)] += 1. / n_steps


        for key in visitation:
            local_pressure[key] = (
                visitation[key]
                * visitation[key]
            )

        for key in visitation:
            self.denoms[key] += local_pressure[key]


        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            rates[step_id] = 1. / self.denoms[(observation, action)]

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

    def learning_rates(self, observations, actions):
        rates = [0. for _ in range(len(observations))]
        visited = []

        n_steps = len(observations)

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            self.denoms[(observation, action)][step_id] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(observation, action)][step_id])
            )
            self.last_update_seen[(observation, action)][step_id] = self.n_updates_elapsed

            visited.append(((observation, action), step_id))


        for key, step_id in visited:
            self.denoms[key][step_id] += 1. / (n_steps ** 2)

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            rates[step_id] = 1. / (self.denoms[(observation, action)][step_id])

        self.n_updates_elapsed += 1
        return rates



class TrajTabularLearningRateScheme():
    def __init__(self, ref_model, has_only_observation_as_key = False, time_horizon = 100.):
        self.denoms = {key: 0. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_observation_as_key = has_only_observation_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_observation_as_key = self.has_only_observation_as_key

        return scheme


    def learning_rates(self, observations, actions):
        rates = [0. for _ in range(len(observations))]

        for observation, action in zip(observations, actions):
            if self.has_only_observation_as_key:
                self.denoms[observation] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[observation])
                )
                self.last_update_seen[observation] = self.n_updates_elapsed

            else:
                self.denoms[(observation, action)] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(observation, action)])
                )
                self.last_update_seen[(observation, action)] = self.n_updates_elapsed

        for observation, action in zip(observations, actions):
            if self.has_only_observation_as_key:
                self.denoms[observation] += 1

            else:
                self.denoms[(observation, action)] += 1

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            if self.has_only_observation_as_key:
                rates[step_id] = 1. / self.denoms[observation]

            else:
                rates[step_id] = 1. / self.denoms[(observation, action)]

        self.n_updates_elapsed += 1
        return rates



class SteppedTabularLearningRateScheme():

    def __init__(self, ref_model, has_only_observation_as_key = False, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_observation_as_key = has_only_observation_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_observation_as_key = self.has_only_observation_as_key

        return scheme

    def learning_rates(self, observations, actions):
        rates = [0. for _ in range(len(observations))]

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            if self.has_only_observation_as_key:
                self.denoms[observation][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[observation][step_id])
                )
                self.last_update_seen[observation][step_id] = self.n_updates_elapsed

            else:
                self.denoms[(observation, action)][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(observation, action)][step_id])
                )
                self.last_update_seen[(observation, action)][step_id] = self.n_updates_elapsed

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            if self.has_only_observation_as_key:
                self.denoms[observation][step_id] += 1

            else:
                self.denoms[(observation, action)][step_id] += 1

        for step_id, (observation, action) in enumerate(zip(observations, actions)):
            if self.has_only_observation_as_key:
                rates[step_id] = 1. / self.denoms[observation][step_id]

            else:
                rates[step_id] = 1. / self.denoms[(observation, action)][step_id]

        self.n_updates_elapsed += 1
        return rates

    # def learning_rates(self, list observations, list actions):
    #     cdef list rates = [0. for _ in range(len(observations))]
    #     cdef Py_ssize_t step_id
    #     cdef double time_horizon =self.time_horizon
    #     cdef double n_updates_elapsed = self.n_updates_elapsed
    #     cdef dict denoms = self.denoms
    #     cdef dict last_update_seen = self.last_update_seen
    #     cdef bint has_only_observation_as_key = self.has_only_observation_as_key
    #
    #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
    #         if has_only_observation_as_key:
    #             denoms[observation][step_id] *= (
    #                 (1. - 1. / time_horizon)
    #                 ** (n_updates_elapsed - last_update_seen[observation][step_id])
    #             )
    #             last_update_seen[observation][step_id] = n_updates_elapsed
    #
    #         else:
    #             denoms[(observation, action)][step_id] *= (
    #                 (1. - 1. / time_horizon)
    #                 ** (n_updates_elapsed - last_update_seen[(observation, action)][step_id])
    #             )
    #             last_update_seen[(observation, action)][step_id] = n_updates_elapsed
    #
    #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
    #         if has_only_observation_as_key:
    #             denoms[observation][step_id] += 1
    #
    #         else:
    #             denoms[(observation, action)][step_id] += 1
    #
    #     for step_id, (observation, action) in enumerate(zip(observations, actions)):
    #         if has_only_observation_as_key:
    #             rates[step_id] = 1. / denoms[observation][step_id]
    #
    #         else:
    #             rates[step_id] = 1. / denoms[(observation, action)][step_id]
    #
    #     n_updates_elapsed += 1
    #     return rates

    def make_light(self, observation_actions_x_episodes):
        light = self.__class__.__new__(self.__class__)

        if self.has_only_observation_as_key:
            light.denoms = make_light_o(self.denoms, observation_actions_x_episodes)
            light.last_update_seen =  make_light_o(self.denoms, observation_actions_x_episodes)
        else:
            light.denoms = make_light(self.denoms, observation_actions_x_episodes)
            light.last_update_seen =  make_light(self.denoms, observation_actions_x_episodes)
        light.n_updates_elapsed = self.n_updates_elapsed
        light.time_horizon = self.time_horizon
        light.has_only_observation_as_key = self.has_only_observation_as_key

        return light


    def update_heavy(self, light):
        update_heavy(self.denoms, light.denoms)
        update_heavy(self.last_update_seen, light.last_update_seen)
        self.n_updates_elapsed = light.n_updates_elapsed
        self.time_horizon = light.time_horizon
        self.has_only_observation_as_key = light.has_only_observation_as_key





class TrajCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = ReducedLearningRateScheme()
        self.core = {key: 0. for key in ref_model}


    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = self.core.copy()

        return critic

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[(observation, action)]
        return evals

    @property
    def time_horizon(self):
        return self.learning_rate_scheme.time_horizon

    @time_horizon.setter
    def time_horizon(self, val):
        self.learning_rate_scheme.time_horizon = val

class SteppedCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.core[(observation, action)]
            evals[step_id] = observation_evals[step_id]
        return evals

    @property
    def time_horizon(self):
        return self.learning_rate_scheme.time_horizon

    @time_horizon.setter
    def time_horizon(self, val):
        self.learning_rate_scheme.time_horizon = val

class HybridCritic():
    def __init__(self, ref_model):
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.stepped_core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}


        self.traj_core = {key: 0. for key in ref_model}

    def copy(self):
        critic = self.__class__(self.stepped_core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.stepped_core = {key : self.stepped_core[key].copy() for key in self.stepped_core.keys()}
        critic.traj_core = self.traj_core.copy()

        return critic

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.traj_core[(observation, action)]

        return evals

    def step_evals_from_stepped(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.stepped_core[(observation, action)]
            evals[step_id] = observation_evals[step_id]
        return evals

    def make_light(self, observation_actions_x_episodes):
        light = self.__class__.__new__(self.__class__)

        light.learning_rate_scheme = self.learning_rate_scheme.make_light(observation_actions_x_episodes)
        light.stepped_core = make_light(self.stepped_core, observation_actions_x_episodes)
        light.traj_core = self.traj_core.copy()

        return light


    def update_heavy(self, light):
        self.learning_rate_scheme.update_heavy(light.learning_rate_scheme)
        update_heavy(self.stepped_core, light.stepped_core)
        self.traj_core = light.traj_core.copy()

    @property
    def time_horizon(self):
        return self.learning_rate_scheme.time_horizon

    @time_horizon.setter
    def time_horizon(self, val):
        self.learning_rate_scheme.time_horizon = val



class AveragedTrajCritic(TrajCritic):
    def eval(self, observations, actions):
        return TrajCritic.eval(self, observations, actions) / len(observations)

class AveragedSteppedCritic(SteppedCritic):
    def eval(self, observations, actions):
        return SteppedCritic.eval(self, observations, actions) / len(observations)

class AveragedHybridCritic(HybridCritic):
    def eval(self, observations, actions):
        return HybridCritic.eval(self, observations, actions) / len(observations)



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

class MidSteppedCritic(AveragedSteppedCritic):

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        fitness = list_sum(rewards)

        estimate = self.eval(observations, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id] / (n_steps * n_steps)
            self.core[(observation, action)][step_id] += delta

class MidHybridCritic(AveragedHybridCritic):

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        fitness = list_sum(rewards)

        estimate = list_sum(self.step_evals_from_stepped(observations, actions)) / n_steps

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id] / (n_steps * n_steps)
            self.stepped_core[(observation, action)][step_id] += delta
            self.traj_core[(observation,action)]  += delta / len(self.stepped_core[(observation, action)])

        # Fully reset Traj Core every so often to address quantization noise.
        if random_uniform() < 0.1 / len(self.traj_core):
            for observation_action in self.traj_core:
                total = 0.
                for stepped_val in self.stepped_core[observation_action]:
                    total += stepped_val
                self.traj_core[observation_action] = total / len(self.stepped_core[observation_action])




class InexactMidTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        AveragedTrajCritic.__init__(self, ref_model)
        self.learning_rate = 1.

    def copy(self):
        critic = AveragedTrajCritic.copy(self)
        critic.learning_rate = self.learning_rate
        return critic

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
            delta = error * learning_rates[step_id] * self.learning_rate
            self.core[(observation, action)] += delta


class InexactMidSteppedCritic(AveragedSteppedCritic):
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
            self.core[(observation, action)][step_id] += delta


class InexactMidHybridCritic(AveragedHybridCritic):
    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        fitness = list_sum(rewards)

        step_evals = self.step_evals_from_stepped(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        for step_id in range(n_steps):
            observation = observations[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.stepped_core[(observation, action)][step_id] += delta
            self.traj_core[(observation, action)] += delta / len(self.stepped_core[(observation, action)])

        # Fully reset Traj Core every so often to address quantization noise.
        if random_uniform() < 0.1 / len(self.traj_core):
            for observation_action in self.traj_core:
                total = 0.
                for stepped_val in self.stepped_core[observation_action]:
                    total += stepped_val
                self.traj_core[observation_action] = total / len(self.stepped_core[observation_action])


class QTrajCritic(AveragedTrajCritic):

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        self.core[(observations[-1], actions[-1])] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[(observations[step_id], actions[step_id])] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )


class QSteppedCritic(AveragedSteppedCritic):
    def __init__(self, ref_model):
        AveragedSteppedCritic.__init__(self, ref_model)
        self.trace_sustain = 0.

    def copy(self):
        critic = AveragedSteppedCritic.copy(self)
        critic.trace_sustain = self.trace_sustain

        return critic

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        deltas = [0.] * n_steps

        deltas[-1] =  rewards[-1] - step_evals[-1]

        for step_id in range(n_steps - 1):
            deltas[step_id] = (
                rewards[step_id]
                + step_evals[step_id + 1]
                - step_evals[step_id]
            )

        apply_eligibility_trace(deltas, self.trace_sustain)

        self.core[(observations[-1], actions[-1])][-1] += learning_rates[-1] * deltas[-1]

        for step_id in range(n_steps - 1):
            observation_action = (observations[step_id], actions[step_id])
            self.core[observation_action][step_id] +=  learning_rates[step_id] * deltas[step_id]





class QHybridCritic(AveragedHybridCritic):
    def __init__(self, ref_model):
        AveragedHybridCritic.__init__(self, ref_model)
        self.trace_sustain = 0.

    def copy(self):
        critic = AveragedHybridCritic.copy(self)
        critic.trace_sustain = self.trace_sustain

        return critic


    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals_from_stepped(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        deltas = [0.] * n_steps

        deltas[-1] =  rewards[-1] - step_evals[-1]

        for step_id in range(n_steps - 1):
            deltas[step_id] = (
                rewards[step_id]
                + step_evals[step_id + 1]
                - step_evals[step_id]
            )

        apply_eligibility_trace(deltas, self.trace_sustain)

        self.stepped_core[(observations[-1], actions[-1])][-1] += learning_rates[-1] * deltas[-1]
        self.traj_core[(observations[-1], actions[-1])] += learning_rates[-1] * deltas[-1] / len(self.stepped_core[(observations[-1], actions[-1])])

        for step_id in range(n_steps - 1):
            observation_action = (observations[step_id], actions[step_id])
            self.stepped_core[observation_action][step_id] +=  learning_rates[step_id] * deltas[step_id]
            self.traj_core[observation_action] +=  learning_rates[step_id] * deltas[step_id] / len(self.stepped_core[observation_action])

        # Fully reset Traj Core every so often to address quantization noise.
        if random_uniform() < 0.1 / len(self.traj_core):
            for observation_action in self.traj_core:
                total = 0.
                for stepped_val in self.stepped_core[observation_action]:
                    total += stepped_val
                self.traj_core[observation_action] = total / len(self.stepped_core[observation_action])

    def make_light(self, observation_actions_x_episodes):
        light = AveragedHybridCritic.make_light(self, observation_actions_x_episodes)
        light.trace_sustain = self.trace_sustain

        return light


    def update_heavy(self, light):
        AveragedHybridCritic.update_heavy(self, light)
        self.trace_sustain = light.trace_sustain


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
        self.core = {key: 0. for key in ref_model}
        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[observation]
        return evals

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)


        self.core[observations[-1]] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[observations[step_id]] += (
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
        self.trace_sustain = 0.


    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}
        critic.trace_sustain = self.trace_sustain

        return critic


    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.core[observation]
            evals[step_id] = observation_evals[step_id]
        return evals

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        deltas = [0.] * n_steps

        deltas[-1] =  rewards[-1] - step_evals[-1]

        for step_id in range(n_steps - 1):
            deltas[step_id] = (
                rewards[step_id]
                + step_evals[step_id + 1]
                - step_evals[step_id]
            )

        apply_eligibility_trace(deltas, self.trace_sustain)

        self.core[observations[-1]][-1] += learning_rates[-1] * deltas[-1]

        for step_id in range(n_steps - 1):
            self.core[observations[step_id]][step_id] +=  learning_rates[step_id] * deltas[step_id]



class VHybridCritic(AveragedHybridCritic):

    def __init__(self, ref_model):
        self.traj_core = {key: 0. for key in ref_model}
        self.stepped_core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.trace_sustain = 0.


    def copy(self):
        critic = self.__class__(self.stepped_core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.stepped_core = {key : self.stepped_core[key].copy() for key in self.stepped_core.keys()}
        critic.traj_core = self.traj_core.copy()
        critic.trace_sustain = self.trace_sustain

        return critic

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.traj_core[observation]
        return evals

    def step_evals_from_stepped(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.stepped_core[observation]
            evals[step_id] = observation_evals[step_id]
        return evals

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals_from_stepped(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        deltas = [0.] * n_steps

        deltas[-1] =  rewards[-1] - step_evals[-1]

        for step_id in range(n_steps - 1):
            deltas[step_id] = (
                rewards[step_id]
                + step_evals[step_id + 1]
                - step_evals[step_id]
            )

        apply_eligibility_trace(deltas, self.trace_sustain)

        self.stepped_core[observations[-1]][-1] += learning_rates[-1] * deltas[-1]
        self.traj_core[observations[-1]] += learning_rates[-1] * deltas[-1] / len(self.stepped_core[observations[-1]])

        for step_id in range(n_steps - 1):
            self.stepped_core[observations[step_id]][step_id] +=  learning_rates[step_id] * deltas[step_id]
            self.traj_core[observations[step_id]] +=  learning_rates[step_id] * deltas[step_id] / len(self.stepped_core[observations[step_id]])


        # Fully reset Traj Core every so often to address quantization noise.
        if random_uniform() < 0.1 / len(self.traj_core):
            for observation in self.traj_core:
                total = 0.
                for stepped_val in self.stepped_core[observation]:
                    total += stepped_val
                self.traj_core[observation] = total / len(self.stepped_core[observation])

class UTrajCritic(AveragedTrajCritic):
    def __init__(self, ref_model):
        self.core = {key: 0. for key in ref_model}
        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[observation]
        return evals

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        self.core[observations[0]] += (
            learning_rates[0]
            * (
                - step_evals[0]
            )
        )

        for step_id in range(1, n_steps):
            self.core[observations[step_id]] += (
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
        self.trace_sustain = 0.

    def copy(self):
        critic = self.__class__(self.core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}
        critic.trace_sustain = self.trace_sustain

        return critic

    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.core[observation]
            evals[step_id] = observation_evals[step_id]
        return evals

    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)

        deltas = [0.] * n_steps

        deltas[0] = -step_evals[0]


        for step_id in range(1, n_steps):
            deltas[step_id] = (
                rewards[step_id - 1]
                + step_evals[step_id - 1]
                - step_evals[step_id]
            )

        apply_reverse_eligibility_trace(deltas, self.trace_sustain)


        self.core[observations[0]][0] += learning_rates[0] * deltas[0]

        for step_id in range(1, n_steps):
            self.core[observations[step_id]][step_id] += learning_rates[step_id] * deltas[step_id]


class UHybridCritic(AveragedHybridCritic):

    def __init__(self, ref_model):
        self.traj_core = {key: 0. for key in ref_model}
        self.stepped_core = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.trace_sustain = 0.


    def copy(self):
        critic = self.__class__(self.stepped_core)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.stepped_core = {key : self.stepped_core[key].copy() for key in self.stepped_core.keys()}
        critic.traj_core = self.traj_core.copy()
        critic.trace_sustain = self.trace_sustain

        return critic




    def step_evals(self, observations, actions):
        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            evals[step_id] = self.traj_core[observation]
        return evals

    def step_evals_from_stepped(self, observations, actions):

        evals = [0. for _ in range(len(observations))]
        for step_id in range(len(observations)):
            observation = observations[step_id]
            action = actions[step_id]
            observation_evals = self.stepped_core[observation]
            evals[step_id] = observation_evals[step_id]
        return evals


    def update(self, observations, actions, rewards):
        n_steps = len(observations)

        step_evals = self.step_evals_from_stepped(observations, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(observations, actions)


        deltas = [0.] * n_steps

        deltas[0] = -step_evals[0]


        for step_id in range(1, n_steps):
            deltas[step_id] = (
                rewards[step_id - 1]
                + step_evals[step_id - 1]
                - step_evals[step_id]
            )

        apply_reverse_eligibility_trace(deltas, self.trace_sustain)


        self.stepped_core[observations[0]][0] += learning_rates[0] * deltas[0]
        self.traj_core[observations[0]] += learning_rates[0] * deltas[0] / len(self.stepped_core[observations[0]])

        for step_id in range(1, n_steps):
            self.stepped_core[observations[step_id]][step_id] += learning_rates[step_id] * deltas[step_id]
            self.traj_core[observations[step_id]] += learning_rates[step_id] * deltas[step_id] / len(self.stepped_core[observations[step_id]])

        # Fully reset Traj Core every so often to address quantization noise.
        if random_uniform() < 0.1 / len(self.traj_core):
            for observation in self.traj_core:
                total = 0.
                for stepped_val in self.stepped_core[observation]:
                    total += stepped_val
                self.traj_core[observation] = total / len(self.stepped_core[observation])

    def make_light(self, observation_actions_x_episodes):
        light = self.__class__.__new__(self.__class__)

        light.learning_rate_scheme = self.learning_rate_scheme.make_light(observation_actions_x_episodes)
        light.stepped_core = make_light_o(self.stepped_core, observation_actions_x_episodes)
        light.traj_core = self.traj_core.copy()

        light.trace_sustain = self.trace_sustain

        return light

    def update_heavy(self, light):
        AveragedHybridCritic.update_heavy(self, light)
        self.trace_sustain = light.trace_sustain




class ABaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, observations, actions, rewards):
        self.v_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions))

    def step_evals(self, observations, actions):
        q_step_evals = self.q_critic.step_evals(observations, actions)
        v_step_evals = self.v_critic.step_evals(observations, actions)
        return [q_step_evals[i] - v_step_evals[i] for i in range(len(q_step_evals))]

    @property
    def time_horizon(self):
        raise NotImplementedError()

    @time_horizon.setter
    def time_horizon(self, val):
        self.v_critic.learning_rate_scheme.time_horizon = val
        self.q_critic.learning_rate_scheme.time_horizon = val

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

class AHybridCritic(ABaseCritic):
    def __init__(self, ref_model_q, ref_model_v):
        self.q_critic = QHybridCritic(ref_model_q)
        self.v_critic = VHybridCritic(ref_model_v)

    def copy(self):
        critic = self.__class__(self.q_critic.stepped_core, self.v_critic.stepped_core)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

class UqBaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, observations, actions, rewards):
        self.u_critic.update(observations, actions, rewards)
        self.q_critic.update(observations, actions, rewards)

    def eval(self, observations, actions):
        return list_sum(self.step_evals(observations, actions)) / len(observations)

    def step_evals(self, observations, actions):
        q_step_evals = self.q_critic.step_evals(observations, actions)
        u_step_evals = self.u_critic.step_evals(observations, actions)
        return [q_step_evals[i] + u_step_evals[i] for i in range(len(q_step_evals))]

    @property
    def time_horizon(self):
        raise NotImplementedError()

    @time_horizon.setter
    def time_horizon(self, val):
        self.u_critic.learning_rate_scheme.time_horizon = val
        self.q_critic.learning_rate_scheme.time_horizon = val

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

class UqHybridCritic(UqBaseCritic):
    def __init__(self, ref_model_q, ref_model_u):
        self.q_critic = QHybridCritic(ref_model_q)
        self.u_critic = UHybridCritic(ref_model_u)

    def copy(self):
        critic = self.__class__(self.q_critic.stepped_core, self.u_critic.stepped_core)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

    def make_light(self, observation_actions_x_episodes):

        light = self.__class__.__new__(self.__class__)

        light.q_critic = self.q_critic.make_light(observation_actions_x_episodes)
        light.u_critic = self.u_critic.make_light(observation_actions_x_episodes)

        return light


    def update_heavy(self, light):
        self.q_critic.update_heavy(light.q_critic)
        self.u_critic.update_heavy(light.u_critic)


    @property
    def trace_sustain(self):
        raise RuntimeError()

    @trace_sustain.setter
    def trace_sustain(self, val):
        a = self.u_critic.trace_sustain
        b = self.q_critic.trace_sustain
        self.u_critic.trace_sustain = val
        self.q_critic.trace_sustain = val
