from multiagent_gridworld import *
from runner import *
from mods import *
import itertools
import random
from multiprocessing import Process
from time import sleep
import sys


def run():
    experiment_name = "MAG_4by4"
    n_stats_run_per_process = 1


    mods_to_mix = [
        (noc,msc,imsc,qsc,asc,uqsc),
    ]


    runners = [
        Runner(experiment_name, setup_combo)
        for setup_combo in itertools.product(*mods_to_mix)
    ]

    random.shuffle(runners)

    for i in range(n_stats_run_per_process):
        for runner in runners:
            runner.new_run()



if __name__ == '__main__':
    # r = Runner('test', (noc,))
    # r.new_run()
    n_processes = int(sys.argv[1])
    print(f"Number of processes: {n_processes}")

    processes = [Process(target = run) for _ in range(n_processes)]

    for process in processes:
        process.start()
        sleep(2)


    for process in processes:
        process.join()
