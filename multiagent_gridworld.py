
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
from critic import *
import random
from min_entropy_dist import my_optimize



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

def observation_cal(posns, goal, agent_id):
    other_agent_id = 1 - agent_id # 0 or 1 if agent is 1 or 0, respectfully.

    agent_diff_0 = posns[other_agent_id][0] -  posns[agent_id][0]
    agent_diff_1 = posns[other_agent_id][1] -  posns[agent_id][1]
    goal_diff_0 = goal[0] - posns[agent_id][0]
    goal_diff_1 = goal[1] - posns[agent_id][1]

    observation = (
        int(agent_diff_0 > 0) - int(agent_diff_0 < 0),
        int(agent_diff_1 > 0) - int(agent_diff_1 < 0),
        int(goal_diff_0 > 0) - int(goal_diff_0 < 0),
        int(goal_diff_1 > 0) - int(goal_diff_1 < 0)
    )

    return observation

def all_observations():
    for agent_diff_0 in range(-1, 2):
        for agent_diff_1 in range(-1, 2):
            for goal_diff_0 in range(-1, 2):
                for goal_diff_1 in range(-1, 2):
                    yield (agent_diff_0, agent_diff_1, goal_diff_0, goal_diff_1)

def all_actions():
    yield Action.LEFT
    yield Action.RIGHT
    yield Action.UP
    yield Action.DOWN
    yield Action.STAY

def get_random_from_list(l):
    r = random_uniform()
    return l[int(r*len(l))]

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
        self.goal = (0,0)
        self.n_agents = 2


    def execute(self, policies):
        n_agents = self.n_agents
        n_steps = self.n_steps
        n_rows = self.n_rows
        n_cols = self.n_cols

        posns = [(int(random_uniform() * n_rows), int(random_uniform() * n_cols)) for i in range(n_agents)]
        self.goal = (int(random_uniform() * n_rows), int(random_uniform() * n_cols))

        trajectories = [Trajectory(n_steps) for i in range(n_agents)]
        rewards = [0. for i in range(self.n_steps)]

        for step_id in range(n_steps):
            actions = [None for i in range(n_agents)]

            ending_early = False

            for agent_id in range(n_agents):
                observation = observation_cal(posns, self.goal, agent_id)

                trajectories[agent_id].observations[step_id] = observation

                possible_actions = (
                    possible_actions_for_cell(posns[agent_id], n_rows, n_cols)
                )

                observation = observation_cal(posns, self.goal, agent_id)
                action = policies[agent_id].action(observation, possible_actions)
                trajectories[agent_id].actions[step_id] = action

                if random_uniform() < self.action_fail_rate:
                    resulting_action = get_random_from_list(possible_actions)
                else:
                    resulting_action = action

                posns[agent_id] = (
                    target_cell_given_action(posns[agent_id], resulting_action, n_rows, n_cols)
                )

            # manhattan_distance = 0
            # for agent_id in range(n_agents):
            #     manhattan_distance += (
            #         abs(posns[agent_id][0] - self.goal[0])
            #         + abs(posns[agent_id][1] - self.goal[1])
            #     )


            reward = -self.time_cost
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
    def __init__(self, dist):
        self.action_probabilities = {}

        for observation in all_observations():
            action_probs = np.random.dirichlet(dist[observation])
            self.action_probabilities[observation] = action_probs

    def copy(self):
        policy = Policy(self.n_rows, self.n_cols)

        policy.action_probabilities = self.action_probabilities.copy()

        return policy


    def action(self, observation, possible_actions):
        r = random_uniform()
        action_probs = self.action_probabilities[observation]
        total_prob = 0.

        for action in possible_actions:
            total_prob += action_probs[action]

        r *= total_prob

        selected_action = possible_actions[0]
        p = 0.
        for action in possible_actions:
            selected_action = action
            p += action_probs[action]
            if p > r:
                break

        return selected_action

    def mutate(self, dist):
        for observation in all_observations():
            self.action_probabilities[observation] = np.random.dirichlet(dist[observation])

def create_dist(n_rows, n_cols):
    dist = {}

    for observation in all_observations():
        dist[observation] =  np.ones(len(list(all_actions())))

    return dist

def update_dist(dist, kl_penalty_factor, phenotypes):
    phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])

    selected_policies = [None] * (len(phenotypes) // 2)

    for i in range(len(selected_policies)):
        selected_policies[i] = phenotypes[i]["policy"]


    for observation in all_observations():
        data = [None] * len(selected_policies)
        for policy_id in range(len(selected_policies)):
            policy = selected_policies[policy_id]
            data[policy_id] = policy.action_probabilities[observation]

        dist[observation] = my_optimize(dist[observation], kl_penalty_factor, data)

# def update_dist(dist, speed, sustain, phenotypes):
#     phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])
#
#     for i in range(len(phenotypes) // 2):
#         policy = phenotypes[i]["policy"]
#
#
#         trajectory = phenotypes[i]["trajectory"]
#
#         observations = trajectory.observations
#         actions = trajectory.actions
#
#         for observation, action in zip(observations, actions):
#             dist[observation][action] += speed
#         # for observation in all_observations():
#         #     action_probabilities = policy.action_probabilities[observation]
#         #     for action in all_actions():
#         #         dist[observation][action] += speed * action_probabilities[action]
#
#     for observation in all_observations():
#         for action in all_actions():
#             dist[observation][action] = (dist[observation][action]  - 1.) * sustain + 1.
#
#     random.shuffle(phenotypes)
#
# def update_dist(dist, speed, sustain, phenotypes):
#     phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])
#
#     for i in range(len(phenotypes) // 2):
#         policy = phenotypes[i]["policy"]
#
#
#         trajectory = phenotypes[i]["trajectory"]
#
#         observations = trajectory.observations
#         actions = trajectory.actions
#
#         for observation, action in zip(observations, actions):
#             dist[observation][action] += speed
#         # for observation in all_observations():
#         #     action_probabilities = policy.action_probabilities[observation]
#         #     for action in all_actions():
#         #         dist[observation][action] += speed * action_probabilities[action]
#
#     for observation in all_observations():
#         for action in all_actions():
#             dist[observation][action] = (dist[observation][action]  - 1.) * sustain + 1.
#
#     random.shuffle(phenotypes)
#



    # phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])
    #
    # for i in range(len(phenotypes) // 2):
    #     policy = phenotypes[i]["policy"]
    #     trajectory = phenotypes[i]["trajectory"]
    #
    #     observations = trajectory.observations
    #     actions = trajectory.actions
    #
    #     for observation, action in zip(observations, actions):
    #         cell = (observation[0], observation[1])
    #         possible_actions = possible_actions_for_cell(cell, n_rows, n_cols)
    #         dist_observation = dist[observation]
    #
    #         if len(dist_observation) != len(possible_actions):
    #             # Something went wrong
    #             raise RuntimeError("Something went wrong")
    #
    #         for action_id in range(len(possible_actions)):
    #             if possible_actions[action_id] == action:
    #                 dist_observation[action_id] += speed
    #
    #             dist_observation[action_id] *= sustain
    #
    # for observation in dist.keys():
    #     dist_observation = dist[observation]
    #     for action_id in range(len(dist_observation)):
    #         dist_observation[action_id] += bonus_mark
