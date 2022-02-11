from numpy.random import default_rng
import numpy as np
from typing import Mapping, Generic, TypeVar, Iterable, Hashable, MutableMapping, Sequence

rng = default_rng()

ObservationT = TypeVar("ObservationT", bound = Hashable)
ActionT = TypeVar("ActionT", bound = Hashable)

# This types maps an observation to an action-sampling distribution, which maps an action to a probability
ActiontoDistributionT = MutableMapping[ActionT, float]
ObservationToDistributionT = Mapping[ObservationT, ActiontoDistributionT]

def array_from_action_probs(
    action_probs: ActiontoDistributionT,
    all_actions: Sequence[ActionT]
) -> np.array:
    return np.array([action_probs[all_actions[action_id]] for action_id in range(len(all_actions))])

def update_action_probs_with_array(
    action_probs: ActiontoDistributionT,
    arr: Sequence[float],
    all_actions: Sequence[ActionT]
):
    for action_id, action in enumerate(all_actions):
        action_probs[action] = arr[action_id]


class Policy(Generic[ObservationT, ActionT]):
    """
        A policy chooses an action based on a probability from a categorical distribution.

        A categorial distribution is like a weighted k-side dice (k is the number of actions).
        A random categorial distribution can be generated from a dirichlet distribution.
        There is one categorial distribution per observation.
    """

    observation_to_multinomial_map: ObservationToDistributionT

    def __init__(self, observation_to_dirichlet_map: ObservationToDistributionT):
        self.observation_to_multinomial_map: ObservationToDistributionT = {}

        for observation, dirichlet in observation_to_dirichlet_map.items():
            actions = dirichlet.keys()
            dirichlet_parameters = list(dirichlet.values())
            action_probs = rng.dirichlet(dirichlet_parameters)
            action_to_prob_map = {action: prob for action,prob in zip(actions, action_probs)}
            self.observation_to_multinomial_map[observation] = action_to_prob_map

    def action(self, observation: ObservationT, possible_actions: Iterable[ActionT]) -> ActionT:
        r = rng.random()
        action_multinomial_probs = self.observation_to_multinomial_map[observation]

        prob_sum = 0.
        for some_action in possible_actions:
            prob_sum += action_multinomial_probs[some_action]

        possible_prob_sum = r * prob_sum

        prob_counter = 0.
        for some_action in possible_actions:
            selected_action = some_action
            prob_counter += action_multinomial_probs[selected_action]
            if prob_counter > possible_prob_sum:
                break

        return selected_action

    def random_reinit(self, observation_dirichlet_map: ObservationToDistributionT):
        self.observation_multinomial_map: ObservationToDistributionT = {}

        for observation in observation_dirichlet_map:
            all_actions = list(observation_dirichlet_map[observation].keys())
            new_multinomial = (
                rng.dirichlet(
                    array_from_action_probs(
                        observation_dirichlet_map[observation],
                        all_actions
                    )
                )
            )
            action_multinomial_probs = {}
            update_action_probs_with_array(action_multinomial_probs, new_multinomial, all_actions)
            # action_multinomial_probs = rng.dirichlet(observation_dirichlet_map[observation])
            self.observation_multinomial_map[observation] = action_multinomial_probs

    # def copy(self):
    #     policy = Policy(self.n_rows, self.n_cols)
    #
    #     policy.moving_action_probabilities = self.moving_action_probabilities.copy()
    #
    #     return policy
    #
    #
    # def moving_action(self, observation, possible_moving_actions):
    #     r = rng.random()
    #     moving_action_probs = self.moving_action_probabilities[observation]
    #     total_prob = 0.
    #
    #     for moving_action in possible_moving_actions:
    #         total_prob += moving_action_probs[moving_action]
    #
    #     r *= total_prob
    #
    #     selected_moving_action = possible_moving_actions[0]
    #     p = 0.
    #     for moving_action in possible_moving_actions:
    #         selected_moving_action = moving_action
    #         p += moving_action_probs[moving_action]
    #         if p > r:
    #             break
    #
    #     return selected_moving_action
    #
    # def mutate(self, dist):
    #     for observation in all_observations():
    #         self.moving_action_probabilities[observation] = np.random.dirichlet(dist[observation])
#
#
# class MovingPolicy():
#     def __init__(self, dist):
#         self.moving_action_probabilities = {}
#
#         for observation in all_observations():
#             moving_action_probs = np.random.dirichlet(dist[observation])
#             self.moving_action_probabilities[observation] = moving_action_probs
#
#     # def copy(self):
#     #     policy = Policy(self.n_rows, self.n_cols)
#     #
#     #     policy.moving_action_probabilities = self.moving_action_probabilities.copy()
#     #
#     #     return policy
#
#
#     def moving_action(self, observation, possible_moving_actions):
#         r = rng.random()
#         moving_action_probs = self.moving_action_probabilities[observation]
#         total_prob = 0.
#
#         for moving_action in possible_moving_actions:
#             total_prob += moving_action_probs[moving_action]
#
#         r *= total_prob
#
#         selected_moving_action = possible_moving_actions[0]
#         p = 0.
#         for moving_action in possible_moving_actions:
#             selected_moving_action = moving_action
#             p += moving_action_probs[moving_action]
#             if p > r:
#                 break
#
#         return selected_moving_action
#
#     def mutate(self, dist):
#         for observation in all_observations():
#             self.moving_action_probabilities[observation] = np.random.dirichlet(dist[observation])
#
# class TargettingPolicy():
#     def __init__(self, dist):
#         self.target_type_probabilities = {}
#
#         for observation in all_observations():
#             target_type_probs = np.random.dirichlet(dist[observation])
#             self.target_type_probabilities[observation] = target_type_probs
#
#     # def copy(self):
#     #     policy = Policy(self.n_rows, self.n_cols)
#     #
#     #     policy.moving_action_probabilities = self.moving_action_probabilities.copy()
#     #
#     #     return policy
#
#
#     def target_type(self, observation):
#         r = rng.random()
#         target_type_probs = self.target_type_probabilities[observation]
#
#
#         selected_target_type = TargetType.CLOSEST_GOAL
#         p = 0.
#         for target_type in all_target_types():
#             selected_target_type = target_type
#             p += target_type_probs[target_type]
#             if p > r:
#                 break
#
#         return selected_target_type
#
#     def mutate(self, dist):
#         for observation in all_observations():
#             self.target_type_probabilities[observation] = np.random.dirichlet(dist[observation])


