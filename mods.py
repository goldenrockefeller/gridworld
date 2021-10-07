from multiagent_gridworld import *

def all_observation_actions():
    for observation in all_observations():
        for action in all_actions():
            yield (observation, action)

def traj_q_model(n_steps):
    model = {observation_action: 0. for observation_action in all_observation_actions()}
    return model

def traj_v_model(n_steps):
    model = {observation: 0. for observation in all_observations()}
    return model


def stepped_q_model(n_steps):
    model = {observation_action: [0.] * n_steps for observation_action in all_observation_actions()}
    return model

def stepped_v_model(n_steps):
    model = {observation: [0.] * n_steps for observation in all_observations()}
    return model


def noc(args):
    args["critic"] =  None

def mtc(args):
    n_steps = args["n_steps"]
    model = traj_q_model(n_steps)
    args["critic"] = MidTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def msc(args):
    n_steps = args["n_steps"]
    model = stepped_q_model(n_steps)
    args["critic"] = MidSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imtc(args):
    n_steps = args["n_steps"]
    model = traj_q_model(n_steps)
    args["critic"] = InexactMidTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imsc(args):
    n_steps = args["n_steps"]
    model = stepped_q_model(n_steps)
    args["critic"] = InexactMidSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qtc(args):
    n_steps = args["n_steps"]
    model = traj_q_model(n_steps)
    args["critic"] = QTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qsc(args):
    n_steps = args["n_steps"]
    model = stepped_q_model(n_steps)
    args["critic"] = QSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def biqtc(args):
    n_steps = args["n_steps"]
    model = traj_q_model(n_steps)
    args["critic"] = BiQTrajCritic(model)
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def biqsc(args):
    n_steps = args["n_steps"]
    model = stepped_q_model(n_steps)
    args["critic"] = BiQSteppedCritic(model)
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def uqtc(args):
    n_steps = args["n_steps"]
    q_model = traj_q_model(n_steps)
    u_model = traj_v_model(n_steps)
    args["critic"] = UqTrajCritic(q_model, u_model)
    args["critic"].u_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def uqsc(args):
    n_steps = args["n_steps"]
    q_model = stepped_q_model(n_steps)
    u_model = stepped_v_model(n_steps)
    args["critic"] = UqSteppedCritic(q_model, u_model)
    args["critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def atc(args):
    n_steps = args["n_steps"]
    q_model = traj_q_model(n_steps)
    v_model = traj_v_model(n_steps)
    args["critic"] = ATrajCritic(q_model, v_model)
    args["critic"].v_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def asc(args):
    n_steps = args["n_steps"]
    q_model = stepped_q_model(n_steps)
    v_model = stepped_v_model(n_steps)
    args["critic"] = ASteppedCritic(q_model, v_model)
    args["critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]