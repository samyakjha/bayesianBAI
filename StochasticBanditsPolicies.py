import numpy as np
import StochasticBanditsModules
import numpy
import pandas
import types


def thompson_sampling_gaussian(arms, horizon):
    ts_environment = StochasticBanditsModules.Environment()
    max_mean_arm = types.SimpleNamespace()

    for arm in arms:
        new_arm = StochasticBanditsModules.GaussianArm(ts_environment, arm.mean, arm.variance)

    mean_arr = pandas.array([])
    for arm in ts_environment.listArms():
        mean_arr.append(arm.mean)

    optimal_mean = np.max(mean_arr)
    regret = pandas.array([])
    for time in range(horizon):
        max_mean = -np.infty

        for arm in ts_environment.listArms():
            if len(arm.history) != 0:
                mean_val = np.sum(arm.history)
                var_val = 1.0/(len(arm.history)+1)
            else:
                mean_val = 0
                var_val = 1.0

            sample_val = numpy.random.normal(mean_val, var_val)
            if sample_val >= max_mean:
                max_mean = sample_val
                max_mean_arm = arm

        max_mean_arm.environment.armPulled.append(max_mean_arm)
        max_mean_arm.environment.gain.append(max_mean)
        max_mean_arm.history.append(max_mean)

        regret.append(optimal_mean - max_mean)

    return ts_environment
