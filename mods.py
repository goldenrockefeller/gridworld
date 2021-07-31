from multiagent_gridworld import *


def none(args):
    args["critic"] = MidTrajCritic()
    args["critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["critic"].core)
    args["n_steps"] = 20
    args["n_rows"] = 3
    args["n_cols"] = 3


def all_observation_actions(n_rows, n_cols):
    for observation in all_observations(n_rows, n_cols):
        yield (observation, Action.LEFT)
        yield (observation, Action.RIGHT)
        yield (observation, Action.UP)
        yield (observation, Action.DOWN)
        yield (observation, Action.STAY)

def traj_q_model(n_steps, n_rows, n_cols):
    model = {observation_action: 0. for observation_action in all_observation_actions(n_rows, n_cols)}
    return model

def traj_v_model(n_steps, n_rows, n_cols):
    model = {observation: 0. for observation in all_observations(n_rows, n_cols)}
    return model


def stepped_q_model(n_steps, n_rows, n_cols):
    model = {observation_action: [0.] * n_steps for observation_action in all_observation_actions(n_rows, n_cols)}
    return model

def stepped_v_model(n_steps, n_rows, n_cols):
    model = {observation: [0.] * n_steps for observation in all_observations(n_rows, n_cols)}
    return model

def noc(args):
    args["critic"] =  None

def mtc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = traj_q_model(n_steps, n_rows, n_cols)
    args["critic"] = MidTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def msc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = stepped_q_model(n_steps, n_rows, n_cols)
    args["critic"] = MidSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imtc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = traj_q_model(n_steps, n_rows, n_cols)
    args["critic"] = InexactMidTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imsc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = stepped_q_model(n_steps, n_rows, n_cols)
    args["critic"] = InexactMidSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qtc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = traj_q_model(n_steps, n_rows, n_cols)
    args["critic"] = QTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qsc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = stepped_q_model(n_steps, n_rows, n_cols)
    args["critic"] = QSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def biqtc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = traj_q_model(n_steps, n_rows, n_cols)
    args["critic"] = BiQTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def biqsc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    model = stepped_q_model(n_steps, n_rows, n_cols)
    args["critic"] = BiQSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def uqtc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    q_model = traj_q_model(n_steps, n_rows, n_cols)
    u_model = traj_v_model(n_steps, n_rows, n_cols)
    args["critic"] = UqTrajCritic(q_model, u_model)
    args["critic"].u_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def uqsc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    q_model = stepped_q_model(n_steps, n_rows, n_cols)
    u_model = stepped_v_model(n_steps, n_rows, n_cols)
    args["critic"] = UqSteppedCritic(q_model, u_model)
    args["critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def atc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    q_model = traj_q_model(n_steps, n_rows, n_cols)
    v_model = traj_v_model(n_steps, n_rows, n_cols)
    args["critic"] = ATrajCritic(q_model, v_model)
    args["critic"].v_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def asc(args):
    n_steps = args["n_steps"]
    n_rows = args["n_rows"]
    n_cols = args["n_cols"]
    q_model = stepped_q_model(n_steps, n_rows, n_cols)
    v_model = stepped_v_model(n_steps, n_rows, n_cols)
    args["critic"] = ASteppedCritic(q_model, v_model)
    args["critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]