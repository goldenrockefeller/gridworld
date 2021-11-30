# cython: profile=True

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


@cython.profile(False)
cpdef double random_uniform():
    global random_uniform_counter, np_random_uniform_cache, random_cache_size

    if random_uniform_counter >= random_cache_size:
        random_uniform_counter = 0
        np_random_uniform_cache = np.random.random(random_cache_size)

    val = np_random_uniform_cache[random_uniform_counter]
    random_uniform_counter += 1
    return val



@cython.profile(False)
cpdef int manhattan_distance(Pos pos_a, Pos pos_b) except *:
    return (
        abs(pos_a.row - pos_b.row)
        + abs(pos_a.col - pos_b.col)
    )


@cython.profile(False)
cpdef Pos closest_goal(Robot robot, list goals):

    cdef Pos the_closest_goal = robot.closest_goal
    cdef Pos pos = robot.pos

    cdef Pos other_pos

    if robot.closest_robots is not None:
        other_pos = robot.closest_robots[0].pos
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


@cython.profile(False)
cpdef list closest_robots(Robot robot, list prev_closest_robots, list robots):

    cdef int sort_n = 3

    if prev_closest_robots is None:
        prev_closest_robots = [robots[i] for i in range(sort_n)]
        for robot_id in range(len(prev_closest_robots)):
            if prev_closest_robots[robot_id] is robot:
                prev_closest_robots[robot_id] = robots[sort_n]

    if robot in prev_closest_robots:
        raise ValueError("robot can not be prev_closest_robots")

    cdef list new_closest_robots = []
    cdef list closest_distances = []


    # Go through each other robot.
    cdef int distance
    cdef Py_ssize_t closest_id
    cdef Robot other_robot
    for other_robot in robots:
        if other_robot is not robot:
            distance = manhattan_distance(robot.pos, other_robot.pos)
            if len(new_closest_robots) < sort_n + 1:
                new_closest_robots.append(other_robot)
                closest_distances.append(distance)

            # bubble sort (do not replace on tie)
            for i in range(len(new_closest_robots) - 1):
                closest_id = len(new_closest_robots) - 2 - i
                if distance < closest_distances[closest_id]:
                    closest_distances[closest_id + 1] = closest_distances[closest_id]
                    new_closest_robots[closest_id + 1] = new_closest_robots[closest_id]
                    closest_distances[closest_id] = distance
                    new_closest_robots[closest_id] = other_robot
                else:
                    break

    # Try not to change the order of previously closest robots if there is a tie.

    for other_robot in reversed(prev_closest_robots):
        distance = manhattan_distance(robot.pos, other_robot.pos)

        # bubble sort (replace on ties)
        for i in range(len(new_closest_robots) - 1):
            closest_id = len(new_closest_robots) - 2 - i
            if distance <= closest_distances[closest_id]:
                closest_distances[closest_id + 1] = closest_distances[closest_id]
                new_closest_robots[closest_id + 1] = new_closest_robots[closest_id]
                closest_distances[closest_id] = distance
                new_closest_robots[closest_id] = other_robot
            else:
                break

    return new_closest_robots[:sort_n]

cpdef tuple observation(Robot robot):
    cdef Pos robot_pos = robot.pos

    cdef int target_type = robot.target_type

    cdef Pos target_pos
    if target_type == TargetType.CLOSEST_GOAL:
        target_pos = robot.closest_goal
    elif target_type == TargetType.CLOSEST_ROBOT_1ST:
        target_pos = robot.closest_robots[0].pos
    elif target_type == TargetType.CLOSEST_ROBOT_2ND:
        target_pos = robot.closest_robots[1].pos
    elif target_type == TargetType.CLOSEST_ROBOT_3RD:
        target_pos = robot.closest_robots[2].pos
    else:
        raise RuntimeError()



    cdef bint is_target_to_the_left
    if target_pos.col < robot_pos.col:
        is_target_to_the_left = True
    elif target_pos.col > robot_pos.col:
        is_target_to_the_left = False
    else:
        is_target_to_the_left = True
        if random_uniform() < 0.5:
            is_target_to_the_left = False

    cdef bint is_target_to_the_top
    if target_pos.row < robot_pos.row:
        is_target_to_the_top = True
    elif target_pos.row > robot_pos.row:
        is_target_to_the_top = False
    else:
        is_target_to_the_top = True
        if random_uniform() < 0.5:
            is_target_to_the_top = False

    cdef int quadrant = is_target_to_the_top * 1 + is_target_to_the_left * 2

    cdef int distance = manhattan_distance(robot_pos, target_pos)

    cdef int distance_state

    if distance == 0:
        distance_state = 0
    elif distance == 1:
        distance_state = 1
    elif distance >= 2 and distance < 4:
        distance_state = 2
    elif distance >= 4 and distance < 8:
        distance_state = 3
    elif distance >= 8:
        distance_state = 4
    else:
        raise RuntimeError()

    return (target_type, quadrant, distance_state)

cpdef list goal_capturers(Pos goal, list robots, Py_ssize_t n_req):
    cdef list capturers = []
    cdef Robot robot
    for robot in robots:
        if manhattan_distance(robot.pos, goal) == 0:
            capturers.append(robot)

        if len(capturers) == n_req:
            return capturers

    return None



class MovingActionEnum:
    def __init__(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.STAY = 4

MovingAction = MovingActionEnum()

class TargetTypeEnum:
    def __init__(self):
        self.CLOSEST_GOAL = 0
        self.CLOSEST_ROBOT_1ST = 1
        self.CLOSEST_ROBOT_2ND = 2
        self.CLOSEST_ROBOT_3RD = 3

TargetType = TargetTypeEnum()


def possible_moving_actions_for_cell(cell, n_rows, n_cols):
    row_id = cell.row
    col_id = cell.col

    possible_moving_actions = [MovingAction.STAY]

    if col_id > 0:
        possible_moving_actions.append(MovingAction.LEFT)

    if col_id < n_cols - 1:
        possible_moving_actions.append(MovingAction.RIGHT)

    if row_id > 0:
        possible_moving_actions.append(MovingAction.UP)

    if row_id < n_rows - 1:
        possible_moving_actions.append(MovingAction.DOWN)

    return possible_moving_actions

def col_direction_from_moving_action(moving_action):
    return (
        (moving_action == MovingAction.LEFT) * -1
        + (moving_action == MovingAction.RIGHT) * 1
    )

def row_direction_from_moving_action(moving_action):
    return (
        (moving_action == MovingAction.UP) * -1
        + (moving_action == MovingAction.DOWN) * 1
    )

def target_cell_given_moving_action(cell, moving_action, n_rows, n_cols):
    row_id = cell.row
    col_id = cell.col

    if moving_action == MovingAction.STAY:
        return Pos(row_id, col_id)

    elif moving_action == MovingAction.LEFT:
        if col_id == 0:
            raise (
                ValueError(
                    f"The moving action LEFT is not a valid moving action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id, col_id - 1)

    elif moving_action == MovingAction.RIGHT:
        if col_id == n_cols - 1:
            raise (
                ValueError(
                    f"The moving action RIGHT is not a valid moving action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id, col_id + 1)

    elif moving_action == MovingAction.UP:
        if row_id == 0:
            raise (
                ValueError(
                    f"The moving action UP is not a valid moving action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return  Pos(row_id - 1, col_id)

    elif moving_action == MovingAction.DOWN:
        if row_id == n_rows - 1:
            raise (
                ValueError(
                    f"The moving action DOWN is not a valid moving action for state "
                    f"{(row_id, col_id)} with boundaries of {(n_rows, n_cols)}"
                )
            )
        else:
            return Pos(row_id + 1, col_id)
    else:
        raise RuntimeError()



def all_observations():
    for target_type in all_target_types():
        for quadrant in range(4):
            for distance_state in range(5):
                yield (target_type, quadrant, distance_state)

def all_moving_actions():
    yield MovingAction.LEFT
    yield MovingAction.RIGHT
    yield MovingAction.UP
    yield MovingAction.DOWN
    yield MovingAction.STAY

def all_target_types():
    yield TargetType.CLOSEST_GOAL
    yield TargetType.CLOSEST_ROBOT_1ST
    yield TargetType.CLOSEST_ROBOT_2ND
    yield TargetType.CLOSEST_ROBOT_3RD

def random_elem_from_list(l):
    r = random_uniform()
    return l[int(r*len(l))]

def random_pos(n_rows, n_cols):
    return Pos(int(random_uniform() * n_rows), int(random_uniform() * n_cols))



class Trajectory:
    def __init__(self, n_steps):
        self.rewards = [0. for i in range(n_steps)]
        self.observations = [None for i in range(n_steps)]
        self.moving_actions = [None for i in range(n_steps)]
        self.target_types = [None for i in range(n_steps)]

cdef class Robot:
    cdef public Pos pos
    cdef public int moving_action
    cdef public tuple observation
    cdef public int target_type

    cdef public double deterioration
    cdef public double obs_fail_rate
    cdef public double moving_action_fail_rate

    cdef public list closest_robots
    cdef public Pos closest_goal


    cdef public double obs_fr_A
    cdef public double obs_fr_B
    cdef public double moving_action_fr_A
    cdef public double moving_action_fr_B

    def __init__(self, random_pos):
        self.obs_fr_A = 0.
        self.obs_fr_B = 1.
        self.moving_action_fr_A = 0.
        self.moving_action_fr_B = 1.

        obs_fr_A = self.obs_fr_A
        obs_fr_B = self.obs_fr_B
        moving_action_fr_A = self.moving_action_fr_A
        moving_action_fr_B = self.moving_action_fr_B

        self.pos = random_pos
        self.moving_action = MovingAction.STAY
        self.target_type = TargetType.CLOSEST_GOAL
        self.observation = None
        self.deterioration = 0.
        self.obs_fail_rate = 1. - 1./(obs_fr_A * self.deterioration + obs_fr_B)
        self.moving_action_fail_rate = 1. - 1./(moving_action_fr_A * self.deterioration + moving_action_fr_B)
        self.closest_robots = None
        self.closest_goal = None

    def deteriorate(self):
        obs_fr_A = self.obs_fr_A
        obs_fr_B = self.obs_fr_B
        moving_action_fr_A = self.moving_action_fr_A
        moving_action_fr_B = self.moving_action_fr_B

        self.deterioration  += 1.
        self.obs_fail_rate = 1. - 1./(obs_fr_A * self.deterioration + obs_fr_B)
        self.moving_action_fail_rate = 1. - 1./(moving_action_fr_A * self.deterioration + moving_action_fr_B)

class RobotRecord:
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
    def __init__(self, n_rows, n_cols, n_steps, n_robots, n_req, n_goals):
        self.n_steps = n_steps
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_robots = n_robots
        self.n_req = n_req
        self.n_goals = n_goals


    def execute(self, moving_policies, targetting_policies):
        n_robots = self.n_robots
        n_goals = self.n_goals
        n_steps = self.n_steps
        n_rows = self.n_rows
        n_cols = self.n_cols
        n_req = self.n_req

        cdef Robot robot

        robots = [Robot(Pos(n_rows//2, n_cols//2)) for i in range(n_robots)]
        goals = [random_pos(n_rows, n_cols) for i in range(n_goals)]

        trajectories = [Trajectory(n_steps) for i in range(n_robots)]
        rewards = [0. for i in range(n_steps)]

        records = {
            "n_rows" : n_rows,
            "n_cols" : n_cols,
            "n_steps" : n_steps,
            "n_robots" : n_robots,
            "n_goals" : n_goals,
            "goal_records" : [GoalRecord(n_steps) for i in range(n_goals)],
            "robot_records" : [RobotRecord(n_steps) for i in range(n_robots)],
        }

        goal_records = records["goal_records"]
        robot_records = records["robot_records"]

        for step_id in range(n_steps):
            for goal, goal_record in zip(goals, goal_records):
                goal_record.rows[step_id] = goal.row
                goal_record.cols[step_id] = goal.col

            for robot, robot_record in zip(robots, robot_records):
                robot_record.rows[step_id] = robot.pos.row
                robot_record.cols[step_id] = robot.pos.col


            for robot in robots:
                robot.closest_goal = closest_goal(robot, goals)
                robot.closest_robots = closest_robots(robot, robot.closest_robots, robots)
                robot.observation = observation(robot)

            for robot, trajectory, moving_policy, targetting_policy in zip(robots, trajectories, moving_policies, targetting_policies):
                trajectory.observations[step_id] = robot.observation

                possible_moving_actions = (
                    possible_moving_actions_for_cell(robot.pos, n_rows, n_cols)
                )

                moving_action = moving_policy.moving_action(robot.observation, possible_moving_actions)
                robot.moving_action = moving_action
                trajectory.moving_actions[step_id] = moving_action

                if random_uniform() < robot.moving_action_fail_rate:
                    resulting_moving_action = random_elem_from_list(possible_moving_actions)
                else:
                    resulting_moving_action = moving_action

                target_type = targetting_policy.target_type(robot.observation)
                robot.target_type = target_type
                trajectory.target_types[step_id] = target_type

                robot.pos = (
                    target_cell_given_moving_action(robot.pos, resulting_moving_action, n_rows, n_cols)
                )

            reward = 0.
            for goal_id, goal in enumerate(goals):
                goal_capturers_ = goal_capturers(goal, robots, n_req)
                if goal_capturers_ is not None:
                    reward += 1.
                    goals[goal_id] = random_pos(n_rows, n_cols)
                    for robot in goal_capturers_:
                        robot.deteriorate()

            rewards[step_id] = reward

            for robot, robot_record in zip(robots, robot_records):
                robot_record.row_directions[step_id] = row_direction_from_moving_action(robot.moving_action)
                robot_record.col_directions[step_id] = col_direction_from_moving_action(robot.moving_action)

        for robot_id in range(n_robots):
            trajectories[robot_id].rewards = rewards.copy()


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


class MovingPolicy():
    def __init__(self, dist):
        self.moving_action_probabilities = {}

        for observation in all_observations():
            moving_action_probs = np.random.dirichlet(dist[observation])
            self.moving_action_probabilities[observation] = moving_action_probs

    # def copy(self):
    #     policy = Policy(self.n_rows, self.n_cols)
    #
    #     policy.moving_action_probabilities = self.moving_action_probabilities.copy()
    #
    #     return policy


    def moving_action(self, observation, possible_moving_actions):
        r = random_uniform()
        moving_action_probs = self.moving_action_probabilities[observation]
        total_prob = 0.

        for moving_action in possible_moving_actions:
            total_prob += moving_action_probs[moving_action]

        r *= total_prob

        selected_moving_action = possible_moving_actions[0]
        p = 0.
        for moving_action in possible_moving_actions:
            selected_moving_action = moving_action
            p += moving_action_probs[moving_action]
            if p > r:
                break

        return selected_moving_action

    def mutate(self, dist):
        for observation in all_observations():
            self.moving_action_probabilities[observation] = np.random.dirichlet(dist[observation])

class TargettingPolicy():
    def __init__(self, dist):
        self.target_type_probabilities = {}

        for observation in all_observations():
            target_type_probs = np.random.dirichlet(dist[observation])
            self.target_type_probabilities[observation] = target_type_probs

    # def copy(self):
    #     policy = Policy(self.n_rows, self.n_cols)
    #
    #     policy.moving_action_probabilities = self.moving_action_probabilities.copy()
    #
    #     return policy


    def target_type(self, observation):
        r = random_uniform()
        target_type_probs = self.target_type_probabilities[observation]


        selected_target_type = TargetType.CLOSEST_GOAL
        p = 0.
        for target_type in all_target_types():
            selected_target_type = target_type
            p += target_type_probs[target_type]
            if p > r:
                break

        return selected_target_type

    def mutate(self, dist):
        for observation in all_observations():
            self.target_type_probabilities[observation] = np.random.dirichlet(dist[observation])

def create_moving_dist():
    dist = {}

    for observation in all_observations():
        dist[observation] =  np.ones(len(list(all_moving_actions())))

    return dist

def create_targetting_dist():
    dist = {}

    for observation in all_observations():
        dist[observation] =  np.ones(len(list(all_target_types())))

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
            if isinstance(policy, MovingPolicy):
                probs = policy.moving_action_probabilities[observation]
            elif isinstance(policy, TargettingPolicy):
                probs = policy.target_type_probabilities[observation]
            else:
                raise RuntimeError()
            data[policy_id] = probs

        result = min_entropy_dist_exp_cg(dist[observation], kl_penalty_factor, data)
        dist[observation] = np.exp(result.x)
