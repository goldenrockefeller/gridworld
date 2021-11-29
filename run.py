from multiagent_gridworld import *
from runner import *
from mods import *
import itertools
import random
from multiprocessing import Process
from time import sleep
import sys
import cProfile, pstats


def run():
    experiment_name = "E1_MAG"
    n_stats_run_per_process = 1

    mods_to_mix = [
        (uqhc_l(500, 0.05), )
    ]

    runners = [
        Runner(experiment_name, setup_combo)
        for setup_combo in itertools.product(*mods_to_mix)
    ]

    random.shuffle(runners)

    for i in range(n_stats_run_per_process):
        for runner in runners:
            runner.new_run()



# uqchc_l(500, 0.05)
# uqchc_l(0, 0.05)
# achc_l(500, 0.05)
# uqhc_l(1, 0.0)
# uqchc_l(1, 0.0)
# uqtc_l(500, 0.05)




if __name__ == '__main__':

    # r = Runner('test', (uqchc_l(1, 0.),))
    #
    #
    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    #
    # r.new_run()
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()

    n_processes = int(sys.argv[1])
    print(f"Number of processes: {n_processes}")

    processes = [Process(target = run) for _ in range(n_processes)]

    for process in processes:
        process.start()
        sleep(2)


    for process in processes:
        process.join()
