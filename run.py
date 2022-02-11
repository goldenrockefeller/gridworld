import pyximport; pyximport.install()
from runner import Runner
from mods import *
import itertools
import random
from multiprocessing import Process
from time import sleep
import sys
import cProfile, pstats

# This work require Cython 3 to be installed
# (I know that it is in alpha as on now Feb 03 2022, but it is the more developed version)

def run():
    experiment_name = "test"
    n_stats_run_per_process = 1

    # Codes
    # noc - no critic
    # tw - Trajectory-wise
    # sw - Step-wise
    # q - SARSA
    # bi - Bidirectional
    # a - Advantage
    # e - Ensemble
    # ce - Combined Ensemble
    # _et - this is using an eligibility trace schedule


    mods_to_mix = [(
        # noc, # No critic,
        # tw, # Trajectory-wise,
        # sw_e, # Step-Wise Ensemble
        # sw, # Step-Wise
        # q_et(1, 0.0001), # SARSA TD($\lambda$)
        # q_e_et(1, 0.0001), # SARSA Ensemble TD($\lambda$)
        # bi_et(1, 0.0), # Bidirectional TD(1) Need Rerun
        # bi_ce_et(1, 0.0), # Bidirectional (Combined) Ensemble TD(1)
        # bi_et(1, 0.0001), # Bidirectional TD($\lambda$) Needs rerun
        # bi_ce_et(1, 0.0001), # Bidirectional (Combined) Ensemble TD($\lambda$) Need rerun(for comparison)
        # bi_ce_et(0, 0.0001), # Bidirectional (Combined) Ensemble TD(0)
        # a_ce_et(0, 0.0001), # Advantage(Combined) Ensemble TD($\lambda$)
    )]

    runners = [
        Runner(experiment_name, setup_combo)
        for setup_combo in itertools.product(*mods_to_mix)
    ]

    random.shuffle(runners)

    for i in range(n_stats_run_per_process):
        for runner in runners:
            runner.new_run()


if __name__ == '__main__':

    r = Runner('test', (bi_ce_et(1, 0.),))


    profiler = cProfile.Profile()
    profiler.enable()


    r.new_run()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    # n_processes = int(sys.argv[1])
    # print(f"Number of processes: {n_processes}")
    #
    # if n_processes == 1:
    #     run()
    # else:
    #     processes = [Process(target = run) for _ in range(n_processes)]
    #
    #     for process in processes:
    #         process.start()
    #         sleep(2)
    #
    #
    #     for process in processes:
    #         process.join()
