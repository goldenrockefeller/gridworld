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
            "n_rows" : 100,
            "n_cols" : 100,
            "horizon" : 100
        }

        for setup_func in self.setup_funcs:
            setup_func(args)

        critic_base = args["critic"]

        n_agents = 10
        n_goals = 30
        n_req = 2


        critics = [(critic_base.copy() if critic_base is not None else None) for i in range(n_agents)]
        n_steps = args["n_steps"]
        n_rows = args["n_rows"]
        n_cols = args["n_cols"]

        domain = Domain(n_rows, n_cols, n_steps, n_agents, n_req, n_goals)


        # n_epochs = 1000
        # n_policies = 50
        #
        #
        # speed = 0.1
        # dist_horizon_factor = 0.1

        n_epochs = 1000
        n_policies = 50

        kl_penalty_factor = 1.

        dists = [create_dist(n_rows, n_cols) for _ in range(n_agents)]

        populations = [[Policy(dists[i]) for _ in range(n_policies)] for i in range(n_agents)]


        n_epochs_elapsed = list(range(1, n_epochs + 1))
        n_training_episodes_elapsed = [n_epochs_elapsed[epoch_id] * n_policies for epoch_id in range(n_epochs)]
        n_training_steps_elapsed = [n_epochs_elapsed[epoch_id] * n_steps * n_policies for epoch_id in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_a_evals =[]
        critic_a_score_losses = []


        for epoch_id in range(n_epochs):

            phenotypes_x_agent = [phenotypes_from_population(populations[i]) for i in range(n_agents)]

            new_critics = [(critics[i].copy() if critics[i] is not None else None) for i in range(n_agents)]

            for phenotype_id in range(len(phenotypes_x_agent[0])):

                phenotypes = [phenotypes_x_agent[agent_id][phenotype_id] for agent_id in range(n_agents)]

                policies = [phenotypes[agent_id]["policy"] for agent_id in range(n_agents)]

                trajectories, records = domain.execute(policies)

                for agent_id in range(n_agents):
                    observations = trajectories[agent_id].observations
                    actions = trajectories[agent_id].actions
                    rewards = trajectories[agent_id].rewards

                    if critics[agent_id] is not None:
                        fitness = critics[agent_id].eval(observations, actions)
                    else:
                        fitness = sum(rewards)
                    if new_critics[agent_id] is not None:
                        new_critics[agent_id].update(observations, actions, rewards)
                    phenotypes[agent_id]["fitness"] = fitness
                    phenotypes[agent_id]["trajectory"] = trajectories[agent_id]

            critics = new_critics

            for dist, phenotypes in zip(dists, phenotypes_x_agent):
                update_dist(dist, kl_penalty_factor, phenotypes)

                phenotypes.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
                for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
                    policy = phenotype["policy"]
                    policy.mutate(dist)
                random.shuffle(phenotypes)


            self.critics = critics
            self.dists = dists




            candidate_policies = [populations[agent_id][0] for agent_id in range(n_agents)]
            trajectories, records = domain.execute(candidate_policies)
            print(f"Score: {sum(trajectories[0].rewards)}, Epoch: {epoch_id}")


            score = sum(trajectories[0].rewards)
            observations = trajectories[0].observations
            actions = trajectories[0].actions

            critic_a = critics[0]
            if critic_a is not None:
                critic_a_eval = critic_a.eval(observations, actions)
                critic_a_score_loss = 0.5 * (critic_a_eval - score) ** 2

            scores.append(score)
            if critic_a is not None:
                critic_a_evals.append(  critic_a.eval(observations, actions) )
                critic_a_score_losses.append( critic_a_score_loss )


            # print(critic.learning_rate_scheme.denoms)
        # end for epoch in range(n_epochs):

        score_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                self.trial_name,
                "score",
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
                self.trial_name,
                "records",
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
            writer.writerow(['n_agents', records['n_agents']])

            for goal_id, goal_record in enumerate(records['goal_records']):
                writer.writerow([])
                writer.writerow(["Goal", goal_id])
                writer.writerow(["rows"] + goal_record.rows)
                writer.writerow(["cols"] + goal_record.cols)

            for agent_id, agent_record in enumerate(records['agent_records']):
                writer.writerow([])
                writer.writerow(["Agent", agent_id])
                writer.writerow(["rows"] + agent_record.rows)
                writer.writerow(["cols"] + agent_record.cols)
                writer.writerow(["row_directions"] + agent_record.row_directions)
                writer.writerow(["col_directions"] + agent_record.col_directions)

        self.stat_runs_completed += 1


