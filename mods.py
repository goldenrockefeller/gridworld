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



from multiagent_gridworld import *

from critic import *

def all_observation_moving_actions():
    for observation in all_observations():
        for moving_action in all_moving_actions():
            yield (observation, moving_action)

def all_observation_target_types():
    for observation in all_observations():
        for target_type in all_target_types():
            yield (observation, target_type)


all_observation_keys = [(observation, None) for observation in all_observations()]
all_moving_keys = list(all_observation_moving_actions())
all_target_keys = list(all_observation_target_types())

# def moving_traj_q_model(n_steps):
#     model = {observation_moving_action: 0. for observation_moving_action in all_observation_moving_actions()}
#     return model
#
# def targetting_traj_q_model(n_steps):
#     model = {observation_target_type: 0. for observation_target_type in all_observation_target_types()}
#     return model
#
# def traj_v_model(n_steps):
#     model = {observation: 0. for observation in all_observations()}
#     return model


# def moving_stepped_q_model(n_steps):
#     model = {observation_moving_action: [0.] * n_steps for observation_moving_action in all_observation_moving_actions()}
#     return model
#
# def targetting_stepped_q_model(n_steps):
#     model = {observation_target_type: [0.] * n_steps for observation_target_type in all_observation_target_types()}
#     return model


# def stepped_v_model(n_steps):
#     model = {observation: [0.] * n_steps for observation in all_observations()}
#     return model


def noc(args):
    args["moving_critic"] =  None
    args["targetting_critic"] =  None

def tw(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = TwCritic(all_moving_keys)
    args["moving_critic"].learning_rate_scheme = MeanTrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

    args["targetting_critic"] = TwCritic(all_target_keys)
    args["targetting_critic"].learning_rate_scheme = MeanTrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

#
# def msc(args):
#     n_steps = args["n_steps"]
#
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = TwSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = TwSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#
# def m_e(args):
#     n_steps = args["n_steps"]
#
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = TwEnsembleCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["moving_critic"].stepped_core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = TwEnsembleCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["targetting_critic"].stepped_core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]


def sw(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = SwCritic(all_moving_keys)
    args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

    args["targetting_critic"] = SwCritic(all_target_keys)
    args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

# def sw_slow(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_traj_q_model(n_steps)
#     args["moving_critic"] = SwCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].learning_rate = 1. / n_steps
#
#     targetting_model = targetting_traj_q_model(n_steps)
#     args["targetting_critic"] = SwCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].learning_rate = 1. / n_steps

#
# def imsc(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = SwSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = SwSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def sw_e(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = SwEnsembleCritic(all_moving_keys, n_steps)
    args["moving_critic"].process_noise = args["process_noise"]

    args["targetting_critic"] = SwEnsembleCritic(all_target_keys, n_steps)
    args["targetting_critic"].process_noise = args["process_noise"]

def q(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = QCritic(all_moving_keys)
    args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    args["targetting_critic"] = QCritic(all_target_keys)
    args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def q_et(a, b):
    def q_et_inner(args):
        q(args)
        args["trace_horizon_hist"] = (a, b)

    q_et_inner.__name__ = f"q_{a:.0f}_{b}"
    return q_et_inner


#
#
# def qsc(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = QSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = QSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def q_e(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = QEnsembleCritic(all_moving_keys, n_steps)
    args["moving_critic"].process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    args["targetting_critic"] = QEnsembleCritic(all_target_keys, n_steps)
    args["targetting_critic"].process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def q_e_et(a, b):
    def q_e_et_inner(args):
        q_e(args)
        args["trace_horizon_hist"] = (a, b)

    q_e_et_inner.__name__ = f"q_e_{a:.0f}_{b}"
    return q_e_et_inner

# def q_e_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def q_e_et_inner(args):
#         q_e(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].trace_sustain = 1.
#             args["targetting_critic"].trace_sustain = 1.
#         else:
#             args["moving_critic"].trace_sustain = trace_sustain
#             args["targetting_critic"].trace_sustain = trace_sustain
#
#     q_e_et_inner.__name__ = f"q_e_{trace_horizon:.0f}"
#     return q_e_et_inner
#
# def qsc_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def qsc_et_inner(args):
#         qsc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].trace_sustain = 1.
#             args["targetting_critic"].trace_sustain = 1.
#         else:
#             args["moving_critic"].trace_sustain = trace_sustain
#             args["targetting_critic"].trace_sustain = trace_sustain
#
#     qsc_et_inner.__name__ = f"qsc_{trace_horizon:.0f}"
#     return qsc_et_inner

# def biq(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_traj_q_model(n_steps)
#     args["moving_critic"] = BiQCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
# def biqsc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
# def biq_e(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def bi(args):
    n_steps = args["n_steps"]




    args["moving_critic"] = BiCritic(all_moving_keys, all_observation_keys)
    args["moving_critic"].u_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].u_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    args["targetting_critic"] = BiCritic(all_target_keys, all_observation_keys)
    args["targetting_critic"].u_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].u_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def bi_et(a, b):
    def bi_et_inner(args):
        bi(args)
        args["trace_horizon_hist"] = (a, b)

    bi_et_inner.__name__ = f"bi_{a:.0f}_{b}"
    return bi_et_inner


# def bisc(args):
#     n_steps = args["n_steps"]
#
#     q_moving_model = moving_stepped_q_model(n_steps)
#     u_moving_model = stepped_v_model(n_steps)
#     args["moving_critic"] = BiSteppedCritic(q_moving_model, u_moving_model)
#     args["moving_critic"].u_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].u_critic.core, True)
#     args["moving_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
#     args["moving_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
#
#     q_targetting_model = targetting_stepped_q_model(n_steps)
#     u_targetting_model = stepped_v_model(n_steps)
#     args["targetting_critic"] = BiSteppedCritic(q_targetting_model, u_targetting_model)
#     args["targetting_critic"].u_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].u_critic.core, True)
#     args["targetting_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
#     args["targetting_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]

def bi_e(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = BiEnsembleCritic(all_moving_keys, all_observation_keys, n_steps)
    args["moving_critic"].u_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    args["targetting_critic"] = BiEnsembleCritic(all_target_keys, all_observation_keys, n_steps)
    args["targetting_critic"].u_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def bi_e_et(a, b):
    def bi_e_et_inner(args):
        bi_e(args)
        args["trace_horizon_hist"] = (a, b)

    bi_e_et_inner.__name__ = f"bi_e_{a:.0f}_{b}"
    return bi_e_et_inner

def bi_ce(args):
    n_steps = args["n_steps"]


    args["moving_critic"] = BiCombinedEnsembleCritic(all_moving_keys, all_observation_keys, n_steps)
    args["moving_critic"].u_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].core.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    args["targetting_critic"] = BiCombinedEnsembleCritic(all_target_keys, all_observation_keys, n_steps)
    args["targetting_critic"].u_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].core.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def bi_ce_et(a, b):
    def bi_ce_et_inner(args):
        bi_ce(args)
        args["trace_horizon_hist"] = (a, b)

    bi_ce_et_inner.__name__ = f"bi_ce_{a:.0f}_{b}"
    return bi_ce_et_inner



#
# def bi_e_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def bi_e_et_inner(args):
#         bi_e(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].u_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].u_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].u_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].u_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     bi_e_et_inner.__name__ = f"bi_e_{trace_horizon:.0f}"
#     return bi_e_et_inner
#
# def bi_e_A(args):
#     bi_e(args)
#
#     trace_sustain_25 = (25 - 1.) / 25
#     trace_sustain_50 = (50 - 1.) / 50
#
#     trace_schedule = [trace_sustain_50] * 2000 + [trace_sustain_25] * 1000
#
#     args["moving_critic"].u_critic.trace_sustain = trace_sustain_50
#     args["moving_critic"].q_critic.trace_sustain = trace_sustain_50
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_sustain_50
#     args["targetting_critic"].q_critic.trace_sustain = trace_sustain_50
#
#     args["trace_schedule"] = trace_schedule
#
#
# def bi_e_B(args):
#     bi_e(args)
#
#     n = 1 / 100
#     trace_horizons = [50 / (1 + n * x) for x in range(3000)]
#
#     trace_schedule = [(trace_horizon - 1.) / (trace_horizon) for trace_horizon in trace_horizons]
#
#     args["moving_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["moving_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["targetting_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["trace_schedule"] = trace_schedule
#
#
# def bi_e_C(args):
#     bi_e(args)
#
#     n = 0.05
#     trace_horizons = [500 / (1 + n * x) for x in range(3000)]
#
#     trace_schedule = [(trace_horizon - 1.) / (trace_horizon) for trace_horizon in trace_horizons]
#
#     args["moving_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["moving_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["targetting_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["trace_schedule"] = trace_schedule


#
# def bi_e_et_no_quant(trace_horizon):
#     f = bi_e_et(trace_horizon)
#     f.__name__ = f"bi_e_{trace_horizon:.0f}_no_quant"
#     return f

#
# def bisc_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def bisc_et_inner(args):
#         bisc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].u_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].u_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].u_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].u_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     bisc_et_inner.__name__ = f"bisc_{trace_horizon:.0f}"
#     return bisc_et_inner




def a(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = ACritic(all_moving_keys, all_observation_keys)
    args["moving_critic"].v_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].v_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    args["targetting_critic"] = ACritic(all_target_keys, all_observation_keys)
    args["targetting_critic"].v_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].v_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def a_et(a, b):
    def a_et_inner(args):
        a(args)
        args["trace_horizon_hist"] = (a, b)

    a_et_inner.__name__ = f"a_{a:.0f}_{b}"
    return a_et_inner

#
# def asc(args):
#     n_steps = args["n_steps"]
#
#     q_moving_model = moving_stepped_q_model(n_steps)
#     v_moving_model = stepped_v_model(n_steps)
#     args["moving_critic"] = ASteppedCritic(q_moving_model, v_moving_model)
#     args["moving_critic"].v_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].v_critic.core, True)
#     args["moving_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
#     args["moving_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
#
#
#     q_targetting_model = targetting_stepped_q_model(n_steps)
#     v_targetting_model = stepped_v_model(n_steps)
#     args["targetting_critic"] = ASteppedCritic(q_targetting_model, v_targetting_model)
#     args["targetting_critic"].v_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].v_critic.core, True)
#     args["targetting_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
#     args["targetting_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]

def a_e(args):
    n_steps = args["n_steps"]

    args["moving_critic"] = AEnsembleCritic(all_moving_keys, all_observation_keys, n_steps)
    args["moving_critic"].v_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    args["targetting_critic"] = AEnsembleCritic(all_target_keys, all_observation_keys, n_steps)
    args["targetting_critic"].v_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def a_e_et(a, b):
    def a_e_et_inner(args):
        a_e(args)
        args["trace_horizon_hist"] = (a, b)

    a_e_et_inner.__name__ = f"a_e_{a:.0f}_{b}"
    return a_e_et_inner

def a_ce(args):
    n_steps = args["n_steps"]


    args["moving_critic"] = ACombinedEnsembleCritic(all_moving_keys, all_observation_keys, n_steps)
    args["moving_critic"].v_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].core.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    args["targetting_critic"] = ACombinedEnsembleCritic(all_target_keys, all_observation_keys, n_steps)
    args["targetting_critic"].v_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].core.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def a_ce_et(a, b):
    def a_ce_et_inner(args):
        a_ce(args)
        args["trace_horizon_hist"] = (a, b)

    a_ce_et_inner.__name__ = f"a_ce_{a:.0f}_{b}"
    return a_ce_et_inner

# def a_e_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def a_e_et_inner(args):
#         a_e(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].v_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].v_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].v_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].v_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     a_e_et_inner.__name__ = f"a_e_{trace_horizon:.0f}"
#     return a_e_et_inner

# def asc_et(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def asc_et_inner(args):
#         asc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].v_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].v_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].v_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].v_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     asc_et_inner.__name__ = f"asc_{trace_horizon:.0f}"
#     return asc_et_inner
