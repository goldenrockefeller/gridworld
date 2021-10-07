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

        for key in visitation:
            self.denoms[key] += local_pressure[key]


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

        n_steps = len(states)

        for step_id, (state, action) in enumerate(zip(states, actions)):
            self.denoms[(state, action)][step_id] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(state, action)][step_id])
            )
            self.last_update_seen[(state, action)][step_id] = self.n_updates_elapsed

            visited.append(((state, action), step_id))


        for key, step_id in visited:
            self.denoms[key][step_id] += 1. / (n_steps ** 2)

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

    @property
    def time_horizon(self):
        return self.learning_rate_scheme.time_horizon

    @time_horizon.setter
    def time_horizon(self, val):
        self.learning_rate_scheme.time_horizon = val

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
