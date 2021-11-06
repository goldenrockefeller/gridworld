
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
from min_entropy_dist import min_entropy_dist_exp_cg, min_entropy_dist_cg

cdef class Pos:
    cdef public int row
    cdef public int col

    def __init__(self, int row, int col):
        self.row = row
        self.col = col


random_cache_size = 10000
random_uniform_counter = 0
np_random_uniform_cache = np.random.random(random_cache_size)
cpdef double random_uniform():
    global random_uniform_counter, np_random_uniform_cache, random_cache_size

    if random_uniform_counter >= random_cache_size:
        random_uniform_counter = 0
        np_random_uniform_cache = np.random.random(random_cache_size)

    val = np_random_uniform_cache[random_uniform_counter]
    random_uniform_counter += 1
    return val


cpdef int manhattan_distance(Pos pos_a, Pos pos_b) except *:
    return (
        abs(pos_a.row - pos_b.row)
        + abs(pos_a.col - pos_b.col)
    )


cpdef Pos closest_goal(Agent agent, list goals):

    cdef Pos the_closest_goal = agent.closest_goal
    cdef Pos pos = agent.pos

    cdef Pos other_pos

    if agent.closest_agents is not None:
        other_pos = agent.closest_agents[0].pos
    else:
        other_pos = pos


    if the_closest_goal is None:
        the_closest_goal  = goals[0]

    # cdef int closest_distance = (
    #     max(
    #         manhattan_distance(pos, the_closest_goal),
    #         manhattan_distance(other_pos, the_closest_goal)
    #     )
    # )
    cdef int closest_distance = manhattan_distance(pos, the_closest_goal)


    cdef Pos goal
    cdef int distance

    for goal in goals:
        # distance = (
        #     max(
        #         manhattan_distance(pos, goal),
        #         manhattan_distance(other_pos, goal)
        #     )
        # )
        distance = manhattan_distance(pos, goal)

        if distance < closest_distance:
            closest_distance = distance
            the_closest_goal  = goal

    return goal


cpdef list closest_agents(Agent agent, list prev_closest_agents, list agents):

    cdef int sort_n = 3

    if prev_closest_agents is None:
        prev_closest_agents = [agents[i] for i in range(sort_n)]
        for agent_id in range(len(prev_closest_agents)):
            if prev_closest_agents[agent_id] is agent:
                prev_closest_agents[agent_id] = agents[sort_n]

    if agent in prev_closest_agents:
        raise ValueError("agent can not be prev_closest_agents")

    cdef list new_closest_agents = []
    cdef list closest_distances = []


    # Go through each other agent.
    cdef int distance
    cdef Py_ssize_t closest_id
    cdef Agent other_agent
    for other_agent in agents:
        if other_agent is not agent:
            distance = manhattan_distance(agent.pos, other_agent.pos)
            if len(new_closest_agents) < sort_n + 1:
                new_closest_agents.append(other_agent)
                closest_distances.append(distance)

            # bubble sort (do not replace on tie)
            for i in range(len(new_closest_agents) - 1):
                closest_id = len(new_closest_agents) - 2 - i
                if distance < closest_distances[closest_id]:
                    closest_distances[closest_id + 1] = closest_distances[closest_id]
                    new_closest_agents[closest_id + 1] = new_closest_agents[closest_id]
                    closest_distances[closest_id] = distance
                    new_closest_agents[closest_id] = other_agent
                else:
                    break

    # Try not to change the order of previously closest agents if there is a tie.

    for other_agent in reversed(prev_closest_agents):
        distance = manhattan_distance(agent.pos, other_agent.pos)

        # bubble sort (replace on ties)
        for i in range(len(new_closest_agents) - 1):
            closest_id = len(new_closest_agents) - 2 - i
            if distance <= closest_distances[closest_id]:
                closest_distances[closest_id + 1] = closest_distances[closest_id]
                new_closest_agents[closest_id + 1] = new_closest_agents[closest_id]
                closest_distances[closest_id] = distance
                new_closest_agents[closest_id] = other_agent
            else:
                break

    return new_closest_agents[:sort_n]

cpdef tuple observation(Agent agent):
    cdef Pos agent_pos = agent.pos
    cdef Pos prev_agent_pos = agent.prev_pos
    cdef Agent other_agent

    cdef int agent_action = agent.action

    cdef double obs_fail_rate = agent.obs_fail_rate

    # cdef list closest_agent_posns = [other_agent.pos for other_agent in agent.closest_agents]
    cdef list prev_closest_agent_posns = [other_agent.prev_pos for other_agent in agent.closest_agents]
    cdef Pos closest_goal = agent.closest_goal

    cdef list closest_agent_flags = [0 for i in range(3)]
    cdef int closest_goal_flag = 0

    for i in range(3):
        if random_uniform() < 1. - obs_fail_rate:
            closest_agent_flags[i] = (
                int(
                    manhattan_distance(agent_pos, prev_closest_agent_posns[i])
                    < manhattan_distance(prev_agent_pos, prev_closest_agent_posns[i])
                )
            )
        else:
            closest_agent_flags[i] = int(random_uniform() < 0.5)



    if random_uniform() < 1. - obs_fail_rate:
        closest_goal_flag = (
            int(
                manhattan_distance(agent_pos, closest_goal)
                < manhattan_distance(prev_agent_pos, closest_goal)
            )
        )
    else:
        closest_goal_flag = int(random_uniform() < 0.5)

    cdef tuple the_observation = (agent_action, *closest_agent_flags, closest_goal_flag)

    return the_observation

cpdef list goal_capturers(Pos goal, list agents, Py_ssize_t n_req):
    cdef list capturers = []
    cdef Agent agent
    for agent in agents:
        if manhattan_distance(agent.pos, goal) == 0:
            capturers.append(agent)

        if len(capturers) == n_req:
            return capturers

    return None



class ActionEnum:
    def __init__(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.STAY = 4

Action = ActionEnum()

def possible_actions_for_cell(cell, n_rows, n_cols):
    row_id = cell.row
    col_id = cell.col

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

def col_direction_from_action(action):
    return (
        (action == Action.LEFT) * -1
        + (action == Action.RIGHT) * 1
    )

def row_direction_from_action(action):
    return (
        (action == Action.UP) * -1
        + (action == Action.DOWN) * 1
    )

def target_cell_given_action(cell, action, n_rows, n_cols):
    row_id = cell.row
    col_id = cell.col

    if action == Action.STAY:
        return Pos(row_id, col_id)

    elif action == Action.LEFT:
        if col_id == 0:
            raise (
                ValueError(
                    f"Action LEFT is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id, col_id - 1)

    elif action == Action.RIGHT:
        if col_id == n_cols - 1:
            raise (
                ValueError(
                    f"Action RIGHT is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id, col_id + 1)

    elif action == Action.UP:
        if row_id == 0:
            raise (
                ValueError(
                    f"Action UP is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return  Pos(row_id - 1, col_id)

    elif action == Action.DOWN:
        if row_id == n_rows - 1:
            raise (
                ValueError(
                    f"Action DOWN is not a valid action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id + 1, col_id)
    else:
        raise RuntimeError()



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


def all_observations():
    for action in all_actions():
        for closest_agent_flag_0 in (0, 1):
            for closest_agent_flag_1 in (0, 1):
                for closest_agent_flag_2 in (0, 1):
                    for closest_goal_flag in (0, 1):
                        yield (action, closest_agent_flag_0, closest_agent_flag_1, closest_agent_flag_2, closest_goal_flag)

def all_actions():
    yield Action.LEFT
    yield Action.RIGHT
    yield Action.UP
    yield Action.DOWN
    yield Action.STAY

def random_elem_from_list(l):
    r = random_uniform()
    return l[int(r*len(l))]

def random_pos(n_rows, n_cols):
    return Pos(int(random_uniform() * n_rows), int(random_uniform() * n_cols))



class Trajectory:
    def __init__(self, n_steps):
        self.rewards = [0. for i in range(n_steps)]
        self.observations = [None for i in range(n_steps)]
        self.actions = [None for i in range(n_steps)]

cdef class Agent:
    cdef public Pos pos
    cdef public Pos prev_pos
    cdef public int action
    cdef public tuple observation
    cdef public double deterioration
    cdef public double obs_fail_rate
    cdef public double action_fail_rate
    cdef public list closest_agents
    cdef public Pos closest_goal


    cdef public double obs_fr_A
    cdef public double obs_fr_B
    cdef public double action_fr_A
    cdef public double action_fr_B

    def __init__(self, random_pos):
        self.obs_fr_A = 0.
        self.obs_fr_B = 1.
        self.action_fr_A = 0.
        self.action_fr_B = 1.

        obs_fr_A = self.obs_fr_A
        obs_fr_B = self.obs_fr_B
        action_fr_A = self.action_fr_A
        action_fr_B = self.action_fr_B

        self.pos = random_pos
        self.prev_pos = random_pos
        self.action = Action.STAY
        self.observation = None
        self.deterioration = 0.
        self.obs_fail_rate = 1. - 1./(obs_fr_A * self.deterioration + obs_fr_B)
        self.action_fail_rate = 1. - 1./(action_fr_A * self.deterioration + action_fr_B)
        self.closest_agents = None
        self.closest_goal = None

    def deteriorate(self):
        obs_fr_A = self.obs_fr_A
        obs_fr_B = self.obs_fr_B
        action_fr_A = self.action_fr_A
        action_fr_B = self.action_fr_B

        self.deterioration  += 1.
        self.obs_fail_rate = 1. - 1./(obs_fr_A * self.deterioration + obs_fr_B)
        self.action_fail_rate = 1. - 1./(action_fr_A * self.deterioration + action_fr_B)

class AgentRecord:
    def __init__(self, n_steps):
        self.rows = [None for i in range(n_steps)]
        self.cols = [None for i in range(n_steps)]
        self.row_directions = [None for i in range(n_steps)]
        self.col_directions = [None for i in range(n_steps)]

class GoalRecord:
    def __init__(self, n_steps):
        self.rows = [None for i in range(n_steps)]
        self.cols = [None for i in range(n_steps)]


class Domain:
    def __init__(self, n_rows, n_cols, n_steps, n_agents, n_req, n_goals):
        self.n_steps = n_steps
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_agents = n_agents
        self.n_req = n_req
        self.n_goals = n_goals


    def execute(self, policies):
        n_agents = self.n_agents
        n_goals = self.n_goals
        n_steps = self.n_steps
        n_rows = self.n_rows
        n_cols = self.n_cols
        n_req = self.n_req

        cdef Agent agent

        agents = [Agent(Pos(n_rows//2, n_cols//2)) for i in range(n_agents)]
        goals = [random_pos(n_rows, n_cols) for i in range(n_goals)]

        trajectories = [Trajectory(n_steps) for i in range(n_agents)]
        rewards = [0. for i in range(n_steps)]

        records = {
            "n_rows" : n_rows,
            "n_cols" : n_cols,
            "n_steps" : n_steps,
            "n_agents" : n_agents,
            "n_goals" : n_goals,
            "goal_records" : [GoalRecord(n_steps) for i in range(n_goals)],
            "agent_records" : [AgentRecord(n_steps) for i in range(n_agents)],
        }

        goal_records = records["goal_records"]
        agent_records = records["agent_records"]

        for step_id in range(n_steps):
            for goal, goal_record in zip(goals, goal_records):
                goal_record.rows[step_id] = goal.row
                goal_record.cols[step_id] = goal.col

            for agent, agent_record in zip(agents, agent_records):
                agent_record.rows[step_id] = agent.pos.row
                agent_record.cols[step_id] = agent.pos.col


            for agent in agents:
                agent.closest_goal = closest_goal(agent, goals)
                agent.closest_agents = closest_agents(agent, agent.closest_agents, agents)
                agent.observation = observation(agent)

            for agent, trajectory, policy in zip(agents, trajectories, policies):
                pos = agent.pos
                agent.prev_pos = pos

                the_observation = agent.observation

                trajectory.observations[step_id] = the_observation

                possible_actions = (
                    possible_actions_for_cell(pos, n_rows, n_cols)
                )

                action = policy.action(the_observation, possible_actions)
                agent.action = action
                trajectory.actions[step_id] = action

                if random_uniform() < agent.action_fail_rate:
                    resulting_action = random_elem_from_list(possible_actions)
                else:
                    resulting_action = action

                agent.pos = (
                    target_cell_given_action(agent.pos, resulting_action, n_rows, n_cols)
                )

            reward = 0.
            for goal_id, goal in enumerate(goals):
                goal_capturers_ = goal_capturers(goal, agents, n_req)
                if goal_capturers_ is not None:
                    reward += 1.
                    goals[goal_id] = random_pos(n_rows, n_cols)
                    for agent in goal_capturers_:
                        agent.deteriorate()

            rewards[step_id] = reward

            for agent, agent_record in zip(agents, agent_records):
                agent_record.row_directions[step_id] = row_direction_from_action(agent.action)
                agent_record.col_directions[step_id] = col_direction_from_action(agent.action)

        for agent_id in range(n_agents):
            trajectories[agent_id].rewards = rewards.copy()


        return trajectories, records


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

        result = min_entropy_dist_exp_cg(dist[observation], kl_penalty_factor, data)
        dist[observation] = np.exp(result.x)

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


