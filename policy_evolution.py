import numpy as np
from policy import Policy, array_from_action_probs, update_action_probs_with_array, ActiontoDistributionT, ObservationToDistributionT
from dirichlet_cross_entropy_opt import min_divergence_dirichlet_exp_cg
from typing import Sequence, List, TypeVar, Mapping, Protocol,MutableMapping, Generic, Hashable, Iterable


ObservationT = TypeVar("ObservationT", bound = Hashable)
ActionT = TypeVar("ActionT", bound = Hashable)
PolicyT = TypeVar("PolicyT", bound = Policy)
ActiontoDistributionT = MutableMapping[ActionT, float]
ObservationToDistributionT = Mapping[ObservationT, ActiontoDistributionT]


class Phenotype(Generic[PolicyT]):
    def __init__(self, policy: Policy):
        self.policy = policy
        self.fitness = 0.


def create_dist(all_observations: Iterable[ActionT], all_actions: Iterable[ActionT]) -> ObservationToDistributionT:
    dist = {}

    for observation in all_observations:
        dist[observation] = {}
        for action in all_actions:
            dist[observation][action] = 1. # A dirichlet distribution (not multinomial)

    return dist

# def create_moving_dist() -> ObservationToDistributionT:
#     dist = {}
#
#     for observation in all_observations():
#         dist[observation] =  np.ones(len(list(all_moving_actions())))
#
#     return dist
#
# def create_targetting_dist() -> ObservationToDistributionT:
#     dist = {}
#
#     for observation in all_observations():
#         dist[observation] =  np.ones(len(list(all_target_types())))
#
#     return dist

def phenotypes_from_policies(policies: Sequence[PolicyT]) -> List[Phenotype[PolicyT]]:
    phenotypes = [None] * len(policies)

    for i in range(len(policies)):
        phenotypes[i] = Phenotype(policies[i])

    return phenotypes

def policies_from_phenotypes(phenotypes: Sequence[Phenotype[PolicyT]]) -> List[PolicyT]:
    policies = [None] * len(phenotypes)

    for i in range(len(phenotypes)):
        policies[i] = phenotypes[i].policy

    return policies



def update_dist(
    dist: ObservationToDistributionT,
    kl_penalty_factor: float,
    phenotypes: Iterable[Phenotype[PolicyT]],
    all_observations: Sequence[ObservationT],
    all_actions: Sequence[ActionT]
):
    sorted_phenotypes = list(phenotypes)
    sorted_phenotypes.sort(reverse = True, key = lambda phenotype : phenotype.fitness)

    elite_policies = [None] * (len(sorted_phenotypes) // 2)

    for i in range(len(elite_policies)):
        elite_policies[i] = sorted_phenotypes[i].policy

    for observation in all_observations:
        data = [None] * len(elite_policies)
        for policy_id in range(len(elite_policies)):
            policy = elite_policies[policy_id]
            action_multinomial_probs = policy.observation_to_multinomial_map[observation]
            data[policy_id] = array_from_action_probs(action_multinomial_probs, all_actions)

        result = (
            min_divergence_dirichlet_exp_cg(
                array_from_action_probs(
                    dist[observation],
                    all_actions
                ),
                kl_penalty_factor,
                data
            )
        )
        update_action_probs_with_array(dist[observation], result, all_actions)
