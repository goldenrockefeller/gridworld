from domain import Domain, all_moving_actions, all_target_types, all_observations
from policy import Policy
from policy_evolution import phenotypes_from_policies, policies_from_phenotypes, create_dist, update_dist
import sys
import errno
import glob
import os
import shutil
import datetime as dt
import csv
import random

class Runner:
    def __init__(self, experiment_name, setup_funcs):
        self.setup_funcs = setup_funcs
        self.stat_runs_completed = 0
        self.experiment_name = experiment_name
        setup_names = []
        for setup_func in setup_funcs:
            setup_names.append(setup_func.__name__)
        self.trial_name = "_".join(setup_names)

        # Create experiment folder if not already created.
        try:
            os.makedirs(os.path.join("log", experiment_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # Save experiment details
        filenames_in_folder = (
            glob.glob("./**.py", recursive = True)
            + glob.glob("./**.pyx", recursive = True)
            + glob.glob("./**.pxd", recursive = True))
        for filename in filenames_in_folder:
            shutil.copy(filename, os.path.join("log", experiment_name, filename))


    def new_run(self):
        datetime_str = (
            dt.datetime.now().isoformat()
            .replace("-", "").replace(':', '').replace(".", "_")
        )

        print(
            "Starting trial.\n"
            f"experiment: {self.experiment_name}\n"
            f"trial: {self.trial_name}\n"
            f"stat run #: {self.stat_runs_completed}\n"
            "datetime: {datetime_str}\n\n"
            .format(**locals()) )
        sys.stdout.flush()

        args = {
            "n_steps" : 100,
            "n_rows" : 10,
            "n_cols" : 10,
            "process_noise" : 1./ 1000.,
            "trace_horizon_hist" : None
        }

        for setup_func in self.setup_funcs:
            setup_func(args)

        moving_critic_template = args["moving_critic_template"]
        targetting_critic_template = args["targetting_critic_template"]

        n_robots = 4 # Each robot is governed by two policies: moving and targetting
        n_goals = 4
        n_req = 2


        moving_critics = [(moving_critic_template.copy() if moving_critic_template is not None else None) for i in range(n_robots)]
        targetting_critics = [(targetting_critic_template.copy() if targetting_critic_template is not None else None) for i in range(n_robots)]
        n_steps = args["n_steps"]
        n_rows = args["n_rows"]
        n_cols = args["n_cols"]


        domain = Domain(n_rows, n_cols, n_steps, n_robots, n_req, n_goals)

        n_epochs = 6000
        n_policies = 50 # per agent policy population (i.e policy size)

        # Each policy for an agent is evaluated with their a separate team every generations.
        n_teams = n_policies # per generation.

        if args["trace_horizon_hist"] is not None:
            a = args["trace_horizon_hist"][0]
            b = args["trace_horizon_hist"][1]

            trace_horizons = [0.] * n_epochs

            for step_id in range(n_epochs):
                x = step_id
                if b * x == 0:
                    trace_horizons[step_id] = float("inf")
                else:
                    trace_horizons[step_id] = (a + b * x) / (b * x)

            trace_schedule = [0.] * len(trace_horizons)
            for step_id, trace_horizon in enumerate(trace_horizons):
                if trace_horizon == float("inf"):
                    trace_schedule[step_id] = 1.
                else:
                    trace_schedule[step_id] = (trace_horizon - 1.) / (trace_horizon)
        else:
            trace_schedule = None


        kl_penalty_factor = 10.

        moving_dists = [create_dist(all_observations, all_moving_actions) for _ in range(n_robots)]
        moving_populations = [[Policy(moving_dists[i]) for _ in range(n_policies)] for i in range(n_robots)]

        targetting_dists = [create_dist(all_observations, all_target_types) for _ in range(n_robots)]
        targetting_populations = [[Policy(targetting_dists[i]) for _ in range(n_policies)] for i in range(n_robots)]


        epochs_elapsed_range = list(range(1, n_epochs + 1))
        training_episodes_elapsed_range = [epochs_elapsed_range[epoch_id] * n_teams for epoch_id in range(n_epochs)]
        training_steps_elapsed_range = [epochs_elapsed_range[epoch_id] * n_steps * n_teams for epoch_id in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_a_evals =[]
        critic_a_score_losses = []

        for epoch_id in range(n_epochs):
            if trace_schedule is not None:
                for robot_id in range(n_robots):
                    if moving_critics[robot_id] is not None:
                        moving_critics[robot_id].trace_sustain = trace_schedule[epoch_id]


                    if targetting_critics[robot_id] is not None:
                        targetting_critics[robot_id].trace_sustain = trace_schedule[epoch_id]


            for population in moving_populations:
                random.shuffle(population)

            for population in targetting_populations:
                random.shuffle(population)

            # These are 2d lists: moving_phenotypes_x_robot[robot_id][phenotype_id] is a phenotype
            moving_phenotypes_x_robot = [phenotypes_from_policies(moving_populations[i]) for i in range(n_robots)]
            targetting_phenotypes_x_robot = [phenotypes_from_policies(targetting_populations[i]) for i in range(n_robots)]

            # These are 2d lists: moving_policies_x_team[team_id][robot_id] is is a policy
            # trajectories_x_team[team_id][robot_id] is the associated trajectory for policy
            # records_x_team[team_id][robot_id] is the associated record for policy
            # This code runs "domain.execute" for all teams
            # See domain module for definitions of trajectories and records.
            moving_policies_x_team = [[moving_phenotypes_x_robot[robot_id][team_id].policy for robot_id in range(n_robots)] for team_id in range(n_teams)]
            targetting_policies_x_team = [[targetting_phenotypes_x_robot[robot_id][team_id].policy for robot_id in range(n_robots)] for team_id in range(n_teams)]
            # ABANDONED args_list = list(zip([domain  for team_id in range(n_policies)], moving_policies_x_team, targetting_policies_x_team))
            trajectories_x_team, domain_records_x_team = (
                zip(
                    *map(
                        lambda args: domain.execute(*args),
                        zip(
                            moving_policies_x_team,
                            targetting_policies_x_team
                        )
                    )
                )
            )


            for team_id in range(n_teams):

                # Get all robot phenotypes, trajectory, and records for the current team
                moving_phenotypes = [moving_phenotypes_x_robot[robot_id][team_id] for robot_id in range(n_robots)]
                targetting_phenotypes = [targetting_phenotypes_x_robot[robot_id][team_id] for robot_id in range(n_robots)]
                trajectories = trajectories_x_team[team_id]
                domain_record = domain_records_x_team[team_id]

                # Update critics (here) BEFORE assigning fitness score (later).
                for robot_id in range(n_robots):
                    observations = trajectories[robot_id].observations
                    moving_actions = trajectories[robot_id].moving_actions
                    target_types = trajectories[robot_id].target_types
                    rewards = trajectories[robot_id].rewards

                    if moving_critics[robot_id] is not None:
                        moving_critics[robot_id].update(observations, moving_actions, rewards)


                    if targetting_critics[robot_id] is not None:
                        targetting_critics[robot_id].update(observations, target_types, rewards)

                # Assign fitness score (here).
                for robot_id in range(n_robots):
                    observations = trajectories[robot_id].observations
                    moving_actions = trajectories[robot_id].moving_actions
                    target_types = trajectories[robot_id].target_types
                    rewards = trajectories[robot_id].rewards

                    if moving_critics[robot_id] is not None:
                        fitness = moving_critics[robot_id].eval(observations, moving_actions)
                    else:
                        fitness = sum(rewards)

                    moving_phenotypes[robot_id].fitness = fitness

                    if targetting_critics[robot_id] is not None:
                        fitness = targetting_critics[robot_id].eval(observations, target_types)
                    else:
                        fitness = sum(rewards)

                    targetting_phenotypes[robot_id].fitness = fitness

            # For each robot, update moving policy-sampling distribution and policy populations
            for dist, phenotypes in zip(moving_dists, moving_phenotypes_x_robot):
                update_dist(dist, kl_penalty_factor, phenotypes, all_observations, all_moving_actions) # dist is modified.

                phenotypes.sort(reverse = False, key = lambda phenotype : phenotype.fitness)
                for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
                    policy = phenotype.policy
                    policy.random_reinit(dist)
                random.shuffle(phenotypes)

            # For each robot, update targetting policy-sampling distribution and policy populations
            for dist, phenotypes in zip(targetting_dists, targetting_phenotypes_x_robot):
                update_dist(dist, kl_penalty_factor, phenotypes, all_observations, all_target_types) # dist is modified.

                phenotypes.sort(reverse = False, key = lambda phenotype : phenotype.fitness)
                for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
                    policy = phenotype.policy
                    policy.random_reinit(dist)
                random.shuffle(phenotypes)

            # Add process noise to fitness critic. This is different from fitness critic update.
            for robot_id in range(n_robots):
                if moving_critics[robot_id] is not None:
                    moving_critics[robot_id].advance_process()

                if targetting_critics[robot_id] is not None:
                    targetting_critics[robot_id].advance_process()

            self.moving_critics = moving_critics
            self.moving_dists = moving_dists

            # Show how well the agents are doing now.
            # Used "sum(trajectories[0].rewards)" since, with this domain, the sampled rewards should be the same for all agents.
            candidate_moving_policies = [moving_populations[robot_id][0] for robot_id in range(n_robots)]
            candidate_targetting_policies = [targetting_populations[robot_id][0] for robot_id in range(n_robots)]
            trajectories, domain_record = domain.execute(candidate_moving_policies, candidate_targetting_policies)
            score = sum(trajectories[0].rewards)
            observations = trajectories[0].observations
            moving_actions = trajectories[0].moving_actions
            print(f"Score: {score}, Epoch: {epoch_id}, Trial: {self.trial_name}")


            # Estimate critic loss (w.r.t to sampled rewards, not expected rewards)
            # (not really a good indicator of critic performance, but can show when things are going horribly wrong)
            critic_a = moving_critics[0]
            if critic_a is not None:
                critic_a_eval = critic_a.eval(observations, moving_actions)
                critic_a_score_loss = 0.5 * (critic_a_eval - score) ** 2

            scores.append(score)
            if critic_a is not None:
                critic_a_evals.append(critic_a.eval(observations, moving_actions))
                critic_a_score_losses.append( critic_a_score_loss )

        # end "for epoch in range(n_epochs)":

        score_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                "score",
                self.trial_name,
                f"score_{datetime_str}.csv"
            )
        )

        # Write performance score curves to file.
        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(score_filename)):
            try:
                os.makedirs(os.path.dirname(score_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(score_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_epochs_elapsed'] + epochs_elapsed_range)
            writer.writerow(['n_training_episodes_elapsed'] + training_episodes_elapsed_range)
            writer.writerow(['n_training_steps_elapsed'] + training_steps_elapsed_range)

            writer.writerow(['scores'] + scores)
            writer.writerow(['critic_a_evals'] + critic_a_evals)
            writer.writerow(['critic_a_score_losses'] + critic_a_score_losses)

        records_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                "records",
                self.trial_name,
                f"records_{datetime_str}.csv"
            )
        )

        # Write agent trajectory for last episode to file.
        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(records_filename)):
            try:
                os.makedirs(os.path.dirname(records_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(records_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_rows', domain_record.n_rows])
            writer.writerow(['n_cols', domain_record.n_cols])
            writer.writerow(['n_steps', domain_record.n_steps])
            writer.writerow(['n_req', domain_record.n_req])
            writer.writerow(['n_goals', domain_record.n_goals])
            writer.writerow(['n_robots', domain_record.n_robots])


            for goal_id, goal_record in enumerate(domain_record.goal_records):
                writer.writerow([])
                writer.writerow(["Goal", goal_id])
                writer.writerow(["rows"] + goal_record.rows)
                writer.writerow(["cols"] + goal_record.cols)

            for robot_id, robot_record in enumerate(domain_record.robot_records):
                writer.writerow([])
                writer.writerow(["Robot", robot_id])
                writer.writerow(["rows"] + robot_record.rows)
                writer.writerow(["cols"] + robot_record.cols)
                writer.writerow(["row_directions"] + robot_record.row_directions)
                writer.writerow(["col_directions"] + robot_record.col_directions)

        self.stat_runs_completed += 1


# ABANDONED CODE
#
# def domain_execute(domain, args):
#     domain = args[0]
#     moving_policies = args[1]
#     targetting_policies = args[2]
#
#     return domain.execute(moving_policies, targetting_policies)
#
# def critic_update(args):
#
#     rewards_x_phenotype = args[0]
#     observations_x_phenotype = args[1]
#     actions_x_phenotype = args[2]
#     critic  = args[3]
#
#     n_phenotypes = len(rewards_x_phenotype)
#     fitnesses = [0.] * n_phenotypes
#
#
#     for phenotype_id in range(n_phenotypes):
#
#         rewards = rewards_x_phenotype[phenotype_id]
#         observations = observations_x_phenotype[phenotype_id]
#         actions = actions_x_phenotype[phenotype_id]
#
#         if critic is not None:
#             fitness = critic.eval(observations, actions)
#         else:
#             fitness = sum(rewards)
#
#         fitnesses[phenotype_id] = fitness
#
#     for phenotype_id in range(n_phenotypes):
#
#         rewards = rewards_x_phenotype[phenotype_id]
#         observations = observations_x_phenotype[phenotype_id]
#         actions = actions_x_phenotype[phenotype_id]
#
#         if critic is not None:
#             critic.update(observations, actions, rewards)
#
#
#     return critic, fitnesses

