from multiagent_gridworld import *
import sys
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
            copy(filename, os.path.join("log", experiment_name, filename))


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
            "horizon" : 1000
        }

        for setup_func in self.setup_funcs:
            setup_func(args)

        moving_critic_base = args["moving_critic"]
        targetting_critic_base = args["targetting_critic"]

        n_robots = 4
        n_goals = 4
        n_req = 2


        moving_critics = [(moving_critic_base.copy() if moving_critic_base is not None else None) for i in range(n_robots)]
        targetting_critics = [(targetting_critic_base.copy() if targetting_critic_base is not None else None) for i in range(n_robots)]
        n_steps = args["n_steps"]
        n_rows = args["n_rows"]
        n_cols = args["n_cols"]

        domain = Domain(n_rows, n_cols, n_steps, n_robots, n_req, n_goals)

        n_epochs = 3000
        n_policies = 50

        kl_penalty_factor = 10.

        moving_dists = [create_moving_dist() for _ in range(n_robots)]
        moving_populations = [[MovingPolicy(moving_dists[i]) for _ in range(n_policies)] for i in range(n_robots)]

        targetting_dists = [create_targetting_dist() for _ in range(n_robots)]
        targetting_populations = [[TargettingPolicy(targetting_dists[i]) for _ in range(n_policies)] for i in range(n_robots)]


        n_epochs_elapsed = list(range(1, n_epochs + 1))
        n_training_episodes_elapsed = [n_epochs_elapsed[epoch_id] * n_policies for epoch_id in range(n_epochs)]
        n_training_steps_elapsed = [n_epochs_elapsed[epoch_id] * n_steps * n_policies for epoch_id in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_a_evals =[]
        critic_a_score_losses = []


        for epoch_id in range(n_epochs):

            for population in moving_populations:
                random.shuffle(population)

            for population in targetting_populations:
                random.shuffle(population)

            moving_phenotypes_x_robot = [phenotypes_from_population(moving_populations[i]) for i in range(n_robots)]
            targetting_phenotypes_x_robot = [phenotypes_from_population(targetting_populations[i]) for i in range(n_robots)]

            new_moving_critics = [(moving_critics[i].copy() if moving_critics[i] is not None else None) for i in range(n_robots)]
            new_targetting_critics = [(targetting_critics[i].copy() if targetting_critics[i] is not None else None) for i in range(n_robots)]

            for phenotype_id in range(len(moving_phenotypes_x_robot[0])):

                moving_phenotypes = [moving_phenotypes_x_robot[robot_id][phenotype_id] for robot_id in range(n_robots)]
                targetting_phenotypes = [targetting_phenotypes_x_robot[robot_id][phenotype_id] for robot_id in range(n_robots)]

                moving_policies = [moving_phenotypes[robot_id]["policy"] for robot_id in range(n_robots)]
                targetting_policies = [targetting_phenotypes[robot_id]["policy"] for robot_id in range(n_robots)]

                trajectories, records = domain.execute(moving_policies, targetting_policies)

                for robot_id in range(n_robots):
                    observations = trajectories[robot_id].observations
                    moving_actions = trajectories[robot_id].moving_actions
                    target_types = trajectories[robot_id].target_types
                    rewards = trajectories[robot_id].rewards

                    if moving_critics[robot_id] is not None:
                        fitness = moving_critics[robot_id].eval(observations, moving_actions)
                    else:
                        fitness = sum(rewards)

                    if new_moving_critics[robot_id] is not None:
                        new_moving_critics[robot_id].update(observations, moving_actions, rewards)

                    moving_phenotypes[robot_id]["fitness"] = fitness

                    if targetting_critics[robot_id] is not None:
                        fitness = targetting_critics[robot_id].eval(observations, target_types)
                    else:
                        fitness = sum(rewards)

                    if new_targetting_critics[robot_id] is not None:
                        new_targetting_critics[robot_id].update(observations, target_types, rewards)

                    targetting_phenotypes[robot_id]["fitness"] = fitness


            moving_critics = new_moving_critics
            targetting_critics = new_targetting_critics

            for dist, phenotypes in zip(moving_dists, moving_phenotypes_x_robot):
                update_dist(dist, kl_penalty_factor, phenotypes)

                phenotypes.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
                for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
                    policy = phenotype["policy"]
                    policy.mutate(dist)
                random.shuffle(phenotypes)

            for dist, phenotypes in zip(targetting_dists, targetting_phenotypes_x_robot):
                update_dist(dist, kl_penalty_factor, phenotypes)

                phenotypes.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
                for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
                    policy = phenotype["policy"]
                    policy.mutate(dist)
                random.shuffle(phenotypes)


            self.moving_critics = moving_critics
            self.moving_dists = moving_dists


            candidate_moving_policies = [moving_populations[robot_id][0] for robot_id in range(n_robots)]
            candidate_targetting_policies = [targetting_populations[robot_id][0] for robot_id in range(n_robots)]
            trajectories, records = domain.execute(candidate_moving_policies, candidate_targetting_policies)
            print(f"Score: {sum(trajectories[0].rewards)}, Epoch: {epoch_id}")


            score = sum(trajectories[0].rewards)
            observations = trajectories[0].observations
            moving_actions = trajectories[0].moving_actions

            critic_a = moving_critics[0]
            if critic_a is not None:
                critic_a_eval = critic_a.eval(observations, moving_actions)
                critic_a_score_loss = 0.5 * (critic_a_eval - score) ** 2

            scores.append(score)
            if critic_a is not None:
                critic_a_evals.append(  critic_a.eval(observations, moving_actions) )
                critic_a_score_losses.append( critic_a_score_loss )


            # print(critic.learning_rate_scheme.denoms)
        # end for epoch in range(n_epochs):

        score_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                "score",
                self.trial_name,
                f"score_{datetime_str}.csv"
            )
        )

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(score_filename)):
            try:
                os.makedirs(os.path.dirname(score_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(score_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_epochs_elapsed'] + n_epochs_elapsed)
            writer.writerow(['n_training_episodes_elapsed'] + n_training_episodes_elapsed)
            writer.writerow(['n_training_steps_elapsed'] + n_training_steps_elapsed)

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

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(records_filename)):
            try:
                os.makedirs(os.path.dirname(records_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(records_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_rows', records['n_rows']])
            writer.writerow(['n_cols', records['n_cols']])
            writer.writerow(['n_steps', records['n_steps']])
            writer.writerow(['n_goals', records['n_goals']])
            writer.writerow(['n_robots', records['n_robots']])

            for goal_id, goal_record in enumerate(records['goal_records']):
                writer.writerow([])
                writer.writerow(["Goal", goal_id])
                writer.writerow(["rows"] + goal_record.rows)
                writer.writerow(["cols"] + goal_record.cols)

            for robot_id, robot_record in enumerate(records['robot_records']):
                writer.writerow([])
                writer.writerow(["Robot", robot_id])
                writer.writerow(["rows"] + robot_record.rows)
                writer.writerow(["cols"] + robot_record.cols)
                writer.writerow(["row_directions"] + robot_record.row_directions)
                writer.writerow(["col_directions"] + robot_record.col_directions)

        self.stat_runs_completed += 1


