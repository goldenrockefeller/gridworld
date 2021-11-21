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

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = MidTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]


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

def imtc_slow(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = InexactMidTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].time_horizon = args["horizon"]
    args["moving_critic"].learning_rate = 1. / n_steps

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = InexactMidTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].time_horizon = args["horizon"]
    args["targetting_critic"].learning_rate = 1. / n_steps


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

def qhc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def qhc_l_inner(args):
        qhc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].trace_sustain = 1.
            args["targetting_critic"].trace_sustain = 1.
        else:
            args["moving_critic"].trace_sustain = trace_sustain
            args["targetting_critic"].trace_sustain = trace_sustain

    qhc_l_inner.__name__ = f"qhc_{trace_horizon:.0f}"
    return qhc_l_inner

def qsc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def qsc_l_inner(args):
        qsc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].trace_sustain = 1.
            args["targetting_critic"].trace_sustain = 1.
        else:
            args["moving_critic"].trace_sustain = trace_sustain
            args["targetting_critic"].trace_sustain = trace_sustain

    qsc_l_inner.__name__ = f"qsc_{trace_horizon:.0f}"
    return qsc_l_inner

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

def uqhc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def uqhc_l_inner(args):
        uqhc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].u_critic.trace_sustain = 1.
            args["moving_critic"].q_critic.trace_sustain = 1.

            args["targetting_critic"].u_critic.trace_sustain = 1.
            args["targetting_critic"].q_critic.trace_sustain = 1.
        else:
            args["moving_critic"].u_critic.trace_sustain = trace_sustain
            args["moving_critic"].q_critic.trace_sustain = trace_sustain

            args["targetting_critic"].u_critic.trace_sustain = trace_sustain
            args["targetting_critic"].q_critic.trace_sustain = trace_sustain

    uqhc_l_inner.__name__ = f"uqhc_{trace_horizon:.0f}"
    return uqhc_l_inner

def uqhc_A(args):
    uqhc(args)

    trace_sustain_25 = (25 - 1.) / 25
    trace_sustain_50 = (50 - 1.) / 50

    trace_schedule = [trace_sustain_50] * 2000 + [trace_sustain_25] * 1000

    args["moving_critic"].u_critic.trace_sustain = trace_sustain_50
    args["moving_critic"].q_critic.trace_sustain = trace_sustain_50

    args["targetting_critic"].u_critic.trace_sustain = trace_sustain_50
    args["targetting_critic"].q_critic.trace_sustain = trace_sustain_50

    args["trace_schedule"] = trace_schedule



def uqhc_l_no_quant(trace_horizon):
    f = uqhc_l(trace_horizon)
    f.__name__ = f"uqhc_{trace_horizon:.0f}_no_quant"
    return f


def uqsc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def uqsc_l_inner(args):
        uqsc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].u_critic.trace_sustain = 1.
            args["moving_critic"].q_critic.trace_sustain = 1.

            args["targetting_critic"].u_critic.trace_sustain = 1.
            args["targetting_critic"].q_critic.trace_sustain = 1.
        else:
            args["moving_critic"].u_critic.trace_sustain = trace_sustain
            args["moving_critic"].q_critic.trace_sustain = trace_sustain

            args["targetting_critic"].u_critic.trace_sustain = trace_sustain
            args["targetting_critic"].q_critic.trace_sustain = trace_sustain

    uqsc_l_inner.__name__ = f"uqsc_{trace_horizon:.0f}"
    return uqsc_l_inner




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

def ahc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def ahc_l_inner(args):
        ahc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].v_critic.trace_sustain = 1.
            args["moving_critic"].q_critic.trace_sustain = 1.

            args["targetting_critic"].v_critic.trace_sustain = 1.
            args["targetting_critic"].q_critic.trace_sustain = 1.
        else:
            args["moving_critic"].v_critic.trace_sustain = trace_sustain
            args["moving_critic"].q_critic.trace_sustain = trace_sustain

            args["targetting_critic"].v_critic.trace_sustain = trace_sustain
            args["targetting_critic"].q_critic.trace_sustain = trace_sustain

    ahc_l_inner.__name__ = f"ahc_{trace_horizon:.0f}"
    return ahc_l_inner

def asc_l(trace_horizon):

    trace_sustain = (trace_horizon - 1.) / trace_horizon

    def asc_l_inner(args):
        asc(args)
        if trace_sustain == float("inf"):
            args["moving_critic"].v_critic.trace_sustain = 1.
            args["moving_critic"].q_critic.trace_sustain = 1.

            args["targetting_critic"].v_critic.trace_sustain = 1.
            args["targetting_critic"].q_critic.trace_sustain = 1.
        else:
            args["moving_critic"].v_critic.trace_sustain = trace_sustain
            args["moving_critic"].q_critic.trace_sustain = trace_sustain

            args["targetting_critic"].v_critic.trace_sustain = trace_sustain
            args["targetting_critic"].q_critic.trace_sustain = trace_sustain

    asc_l_inner.__name__ = f"asc_{trace_horizon:.0f}"
    return asc_l_inner
