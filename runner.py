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
            "n_steps" : 20,
            "n_rows" : 3,
            "n_cols" : 3,
            "horizon" : 1000
        }

        for setup_func in self.setup_funcs:
            setup_func(args)

        critic_a = args["critic"]
        critic_b = critic_a.copy()
        n_steps = args["n_steps"]
        n_rows = args["n_rows"]
        n_cols = args["n_cols"]
        action_fail_rate = 0.0
        time_cost = 0.01
        reward_goal = 1.0

        domain = Domain(n_rows, n_cols, action_fail_rate, time_cost, reward_goal)
        domain.n_steps = n_steps

        n_epochs = 3000
        n_policies = 50


        speed = 0.001
        sustain = 0.9999
        precision = 0.1
        dist_a = create_dist(n_rows, n_cols, precision)
        dist_b = create_dist(n_rows, n_cols, precision)

        population_a = [Policy(n_rows, n_cols) for _ in range(n_policies)]
        population_b = [Policy(n_rows, n_cols) for _ in range(n_policies)]

        n_epochs_elapsed = list(range(1, n_epochs + 1))
        n_training_episodes_elapsed = [n_epochs_elapsed[epoch_id] * n_policies for epoch_id in range(n_epochs)]
        n_training_steps_elapsed = [n_epochs_elapsed[epoch_id] * n_steps * n_policies for epoch_id in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_a_evals = []
        critic_a_score_losses = []


        for epoch_id in range(n_epochs):

            phenotypes_a = phenotypes_from_population(population_a)
            phenotypes_b = phenotypes_from_population(population_b)

            new_critic_a = critic_a.copy()
            new_critic_b = critic_b.copy()

            for phenotype_id in range(len(phenotypes_a)):
                phenotype_a = phenotypes_a[phenotype_id]
                phenotype_b = phenotypes_b[phenotype_id]

                policy_a = phenotype_a["policy"]
                policy_b = phenotype_b["policy"]

                trajectories = domain.execute([policy_a, policy_b])

                observations_a = trajectories[0].observations
                actions_a = trajectories[0].actions
                reward_a = trajectories[0].rewards
                fitness_a = critic_a.eval(observations_a, actions_a)
                new_critic_a.update(observations_a, actions_a, reward_a)
                phenotype_a["fitness"] = fitness_a
                phenotype_a["trajectory"] = trajectories[0]

                observations_b = trajectories[1].observations
                actions_b = trajectories[1].actions
                reward_b = trajectories[1].rewards
                fitness_b = critic_b.eval(observations_b, actions_b)
                new_critic_b.update(observations_b, actions_b, reward_b)
                phenotype_b["fitness"] = fitness_b
                phenotype_b["trajectory"] = trajectories[1]


            critic_a = new_critic_a
            critic_b = new_critic_b

            update_dist(dist_a, speed, sustain, phenotypes_a, n_rows, n_cols)
            update_dist(dist_b, speed, sustain, phenotypes_b, n_rows, n_cols)

            self.critic_a = critic_a
            self.dist_a = dist_a

            phenotypes_a.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
            for phenotype in phenotypes_a[0: 3 * len(phenotypes_b)//4]:
                policy = phenotype["policy"]
                policy.mutate(dist_a)
            random.shuffle(phenotypes_a)

            phenotypes_b.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
            for phenotype in phenotypes_b[0: 3 * len(phenotypes_b)//4]:
                policy = phenotype["policy"]
                policy.mutate(dist_b)
            random.shuffle(phenotypes_b)


            candidate_policy_a = population_a[0]
            candidate_policy_b = population_b[0]
            trajectories = domain.execute([candidate_policy_a, candidate_policy_b])
            print(f"Score: {sum(trajectories[0].rewards)}, Epoch: {epoch_id}")


            score = sum(trajectories[0].rewards)
            observations = list(filter(lambda x: x is not None, trajectories[0].observations))
            actions = list(filter(lambda x: x is not None, trajectories[0].actions))
            critic_a_eval = critic_a.eval(observations, actions)
            critic_a_score_loss = 0.5 * (critic_a_eval - score) ** 2

            scores.append(score)
            critic_a_evals.append(  critic_a.eval(observations, actions) )
            critic_a_score_losses.append( critic_a_score_loss )


            # print(critic.learning_rate_scheme.denoms)
        # end for epoch in range(n_epochs):

        save_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                self.trial_name,
                f"record_{datetime_str}.csv"
            )
        )

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_epochs_elapsed'] + n_epochs_elapsed)
            writer.writerow(['n_training_episodes_elapsed'] + n_training_episodes_elapsed)
            writer.writerow(['n_training_steps_elapsed'] + n_training_steps_elapsed)

            writer.writerow(['scores'] + scores)
            writer.writerow(['critic_a_evals'] + critic_a_evals)
            writer.writerow(['critic_a_score_losses'] + critic_a_score_losses)


        self.stat_runs_completed += 1


