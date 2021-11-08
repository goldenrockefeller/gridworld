from multiagent_gridworld import *

def all_observation_moving_actions():
    for observation in all_observations():
        for moving_action in all_moving_actions():
            yield (observation, moving_action)

def all_observation_target_types():
    for observation in all_observations():
        for target_type in all_target_types():
            yield (observation, target_type)

def moving_traj_q_model(n_steps):
    model = {observation_moving_action: 0. for observation_moving_action in all_observation_moving_actions()}
    return model

def targetting_traj_q_model(n_steps):
    model = {observation_target_type: 0. for observation_target_type in all_observation_target_types()}
    return model

def traj_v_model(n_steps):
    model = {observation: 0. for observation in all_observations()}
    return model


def moving_stepped_q_model(n_steps):
    model = {observation_moving_action: [0.] * n_steps for observation_moving_action in all_observation_moving_actions()}
    return model

def targetting_stepped_q_model(n_steps):
    model = {observation_target_type: [0.] * n_steps for observation_target_type in all_observation_target_types()}
    return model


def stepped_v_model(n_steps):
    model = {observation: [0.] * n_steps for observation in all_observations()}
    return model


def noc(args):
    args["moving_critic"] =  None
    args["targetting_critic"] =  None

def mtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = MidTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    taregtting_model = targettting_traj_q_model(n_steps)
    args["targettting_critic"] = MidTrajCritic(targettting_model)
    args["targettting_critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["targettting_critic"].core)
    args["targettting_critic"].time_horizon = args["horizon"]


def msc(args):
    n_steps = args["n_steps"]


    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = MidSteppedCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = MidSteppedCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]


def mhc(args):
    n_steps = args["n_steps"]


    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = MidHybridCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["moving_critic"].stepped_core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = MidHybridCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["targetting_critic"].stepped_core)
    args["targetting_critic"].time_horizon = args["horizon"]


def imtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = InexactMidTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = InexactMidTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]

def imsc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = InexactMidSteppedCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = InexactMidSteppedCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]

def imhc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = InexactMidHybridCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].stepped_core)
    args["moving_critic"].time_horizon = args["horizon"]


    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = InexactMidHybridCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].stepped_core)
    args["targetting_critic"].time_horizon = args["horizon"]


def qtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = QTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = QTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]


def qsc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = QSteppedCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = QSteppedCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]

def qhc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = QHybridCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].stepped_core)
    args["moving_critic"].time_horizon = args["horizon"]

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = QHybridCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].stepped_core)
    args["targetting_critic"].time_horizon = args["horizon"]


# def biqtc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_traj_q_model(n_steps)
#     args["moving_critic"] = BiQTrajCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].time_horizon = args["horizon"]
#
# def biqsc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].time_horizon = args["horizon"]
#
# def biqhc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].time_horizon = args["horizon"]

def uqtc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_traj_q_model(n_steps)
    u_moving_model = traj_v_model(n_steps)
    args["moving_critic"] = UqTrajCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].u_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].u_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]


    q_targetting_model = targetting_traj_q_model(n_steps)
    u_targetting_model = traj_v_model(n_steps)
    args["targetting_critic"] = UqTrajCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].u_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].u_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]

def uqsc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    u_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = UqSteppedCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].u_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].u_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]

    q_targetting_model = targetting_stepped_q_model(n_steps)
    u_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = UqSteppedCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].u_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].u_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]

def uqhc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    u_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = UqHybridCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].u_critic.stepped_core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].q_critic.stepped_core)
    args["moving_critic"].u_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]

    q_targetting_model = targetting_stepped_q_model(n_steps)
    u_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = UqHybridCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].u_critic.stepped_core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].q_critic.stepped_core)
    args["targetting_critic"].u_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]


def atc(args):
    n_steps = args["n_steps"]

    q_moving_model = traj_q_model(n_steps)
    v_moving_model = traj_v_model(n_steps)
    args["moving_critic"] = ATrajCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].v_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].v_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]

    q_targetting_model = traj_q_model(n_steps)
    v_targetting_model = traj_v_model(n_steps)
    args["targetting_critic"] = ATrajCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].v_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].v_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]

def asc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    v_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = ASteppedCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].v_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].v_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]


    q_targetting_model = targetting_stepped_q_model(n_steps)
    v_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = ASteppedCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].v_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].v_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]

def ahc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    v_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = AHybridCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].v_critic.stepped_core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["moving_critic"].q_critic.stepped_core)
    args["moving_critic"].v_critic.time_horizon = args["horizon"]
    args["moving_critic"].q_critic.time_horizon = args["horizon"]


    q_targetting_model = targetting_stepped_q_model(n_steps)
    v_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = AHybridCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].v_critic.stepped_core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["targetting_critic"].q_critic.stepped_core)
    args["targetting_critic"].v_critic.time_horizon = args["horizon"]
    args["targetting_critic"].q_critic.time_horizon = args["horizon"]