import StochasticBanditsModules
import numpy
import pandas


def thompson_sampling_gaussian(arms, horizon):
    ts_environment = StochasticBanditsModules.Environment()
    prior_mean = pandas.array([])
    prior_variance = pandas.array([])
    sample = pandas.array([])

    for arm in arms:
        new_arm = StochasticBanditsModules.GaussianArm(ts_environment, arm.mean, arm.variance)
        prior_mean.append(arm.mean)
        prior_variance.append(arm.variance)

    for time in range(horizon):




