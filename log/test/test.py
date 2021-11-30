from math import sqrt

def apply_reverse_eligibility_trace(rewards, values, trace_sustain):
    deltas = [0. for i in range(len(rewards))]

    for step_id in range(len(rewards)):
        if step_id == 0:
            deltas[step_id] = -values[step_id]
        else:
            deltas[step_id] = rewards[step_id - 1] + values[step_id - 1] - values[step_id]


    trace = 0.
    for step_id in range(len(deltas)):
        trace += deltas[step_id]
        deltas[step_id]  = trace
        trace *= trace_sustain

    return deltas

def apply_reverse_eligibility_trace_2(rewards, values, trace_sustain):
    targets = [0. for i in range(len(rewards))]

    for step_id in range(len(rewards)):
        if step_id == 0:
            targets[step_id] = 0.
        else:
            targets[step_id] = (
                (1 - trace_sustain) *(rewards[step_id - 1] + values[step_id - 1])
                + trace_sustain * (rewards[step_id - 1] + targets[step_id-1])
            )


    deltas = [targets[i] - values[i] for i in range(len(rewards))]

    return deltas


def combine_targets(targets, target_uncertainties, values, uncertainties):
    n_steps = len(targets)
    new_values = values.copy()
    new_uncertainties = uncertainties.copy()

    for i in range(n_steps):
        if uncertainties[i] == float("inf"):
            new_values[i] = targets[i]
            new_uncertainties[i] = target_uncertainties[i]

        else:
            tw = 1. / target_uncertainties[i]
            vw = 1. / uncertainties[i]
            new_values[i] = (tw * targets[i] + vw * values[i]) / (tw + vw)
            new_uncertainties[i] =(
                (tw / (tw + vw))**2 * target_uncertainties[i]
                + (vw / (tw + vw))**2 * uncertainties[i]
            )

    return new_values, new_uncertainties

def apply_smart_reverse_eligibility_trace(rewards, values, uncertainties):
    trace = 0.
    trace_uncertainty = 0.
    n_steps = len(rewards)
    targets = [ 0. for i in range(n_steps)]
    targets_uncertainties = [ 0. for i in range(n_steps)]
    r_err = get_r_err(n_steps, 1.)
    R_err = 0.
    V_err = 0.
    for step_id in range(n_steps):
        if step_id == 0:
            trace = 0.
            trace_uncertainty = 0.
            mul = 0.
        else:
            v_err = uncertainties[step_id - 1]
            if R_err == 0:
                long_ratio = 1.
                short_ratio = 0.
            else:
                long_uncertainty =  R_err+ V_err + 2 * sqrt(R_err * r_err)
                long_weight = 1. / long_uncertainty
                if v_err == float("inf"):
                    short_uncertainty = float("inf")
                    short_weight = 0.
                else:
                    short_uncertainty = v_err + V_err + 2 * sqrt(V_err * v_err)
                    short_weight = 1./short_uncertainty
                total_weight = long_weight + short_weight

                long_ratio = long_weight / total_weight
                short_ratio = short_weight/total_weight

            trace = rewards[step_id - 1] + long_ratio * trace + short_ratio * values[step_id - 1]


            V_err *= short_ratio * short_ratio
            R_err = r_err + R_err + 2 * sqrt(R_err * r_err)
            trace_uncertainty = R_err + V_err


        targets[step_id] = trace
        if step_id == 0.:
            targets_uncertainties[step_id] = 1 / n_steps
        else:
            targets_uncertainties[step_id] = trace_uncertainty

    return targets, targets_uncertainties

def regular_averaging(new_value, new_weight, e_value, e_weight):
    e_value = (e_weight * e_value + new_value * new_weight) / (new_weight + e_weight)
    e_weight += new_weight

    return e_value, e_weight


def ensemble_averaging(new_value, new_weight, id, e_value, e_weight, values, weights):
    new_values = values.copy()
    new_weights = weights.copy()

    new_values[id] = (new_values[id] * new_weights[id] + new_value * new_weight) / (new_weights[id] + new_weight)
    new_weights[id] += new_weight

    e_value = 0.
    e_weight = 0.

    for i in range(len(values)):
        e_value += new_values[i] * new_weights[i]
        e_weight += new_weights[i]

    e_value /= e_weight

    return e_value, e_weight, new_values, new_weights

def fast_ensemble_averaging(new_value, new_weight, id, e_value, e_weight, values, weights):
    new_values = values.copy()
    new_weights = weights.copy()

    new_values[id] = (new_values[id] * new_weights[id] + new_value * new_weight) / (new_weights[id] + new_weight)
    new_weights[id] += new_weight


    e_value = (
        (e_value * e_weight - weights[id] * values[id] + new_weights[id] * new_values[id])
        / (e_weight - weights[id]  + new_weights[id])
    )
    e_weight = e_weight - weights[id]  + new_weights[id]


    return e_value, e_weight, new_values, new_weights



e_value = 0.; e_weight = 0.; values = [0.] * 10; weights = [0.] * 10;
e_value, e_weight = regular_averaging(5, 5, e_value, e_weight); print(e_value, e_weight)
e_value, e_weight = regular_averaging(10, 2,  e_value, e_weight); print(e_value, e_weight)
e_value, e_weight= regular_averaging(20, 3, e_value, e_weight); print(e_value, e_weight)


e_value = 0.; e_weight = 0.; values = [0.] * 10; weights = [0.] * 10;
e_value, e_weight, values, weights = fast_ensemble_averaging(5, 5, 0, e_value, e_weight, values, weights); print(e_value, e_weight)
e_value, e_weight, values, weights = fast_ensemble_averaging(10, 2, 1, e_value, e_weight, values, weights); print(e_value, e_weight)
e_value, e_weight, values, weights = fast_ensemble_averaging(20, 2, 0, e_value, e_weight, values, weights); print(e_value, e_weight)
e_value, e_weight, values, weights = fast_ensemble_averaging(20, 1, 3, e_value, e_weight, values, weights); print(e_value, e_weight)