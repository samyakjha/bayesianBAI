import StochasticBanditsModules
import numpy
from tqdm import tqdm


# Thompson Sampling with Gaussian Arms

def thompson_sampling_gaussian(arms, horizon):
    ts_environment = StochasticBanditsModules.Environment()

    for arm in arms:
        new_arm = StochasticBanditsModules.GaussianArm(ts_environment, arm.mean, arm.variance_square)

    mean_arr = numpy.array([])

    for arm in ts_environment.listArms():
        mean_arr = numpy.append(mean_arr, arm.mean)

    try:
        optimal_mean = numpy.amax(mean_arr)
    except ValueError:
        optimal_mean = -numpy.infty

    for time in tqdm(range(horizon)):
        max_mean = -numpy.infty
        max_mean_arm = StochasticBanditsModules.GaussianArm(ts_environment, -numpy.infty, variance_square=0, silent=True)

        for arm in ts_environment.listArms():
            if len(arm.history) != 0:
                mean_val = numpy.sum(arm.history) / len(arm.history)
                var_val = 1.0 / (len(arm.history) + 1)
            else:
                mean_val = 0
                var_val = 1.0

            sample_val = numpy.random.normal(mean_val, var_val)
            if sample_val >= max_mean:
                max_mean = sample_val
                max_mean_arm = arm

        # Error here: max_mean_arm has to be manually assigned the value

        pull_val = max_mean_arm.pullArm()
        max_mean_arm.environment.armPulled = numpy.append(max_mean_arm.environment.armPulled, max_mean_arm)
        max_mean_arm.environment.gain = numpy.append(max_mean_arm.environment.gain, pull_val)
        max_mean_arm.history = numpy.append(max_mean_arm.history, pull_val)

        ts_environment.regret = numpy.append(ts_environment.regret, optimal_mean - max_mean)
        print("Time: " + str(time) + " Regret: " + str(ts_environment.regret[time]))

    return ts_environment


# Error here: COMPUTATIONALLY EXPENSIVE!!
# Top Two Thompson Sampling with Gaussian Arms

def ttts_gaussian(arms, horizon, sampling=0.5):
    ttts_environment = StochasticBanditsModules.Environment()

    for arm in arms:
        new_arm = StochasticBanditsModules.GaussianArm(ttts_environment, arm.mean, arm.variance_square)

    mean_arr = numpy.array([])

    for arm in ttts_environment.listArms():
        mean_arr = numpy.append(mean_arr, arm.mean)

    try:
        optimal_mean = numpy.amax(mean_arr)
    except ValueError:
        optimal_mean = -numpy.infty

    max_mean_arm = StochasticBanditsModules.GaussianArm(ttts_environment, -numpy.infty, variance_square=0, silent=True)
    second_max_mean_arm = StochasticBanditsModules.GaussianArm(ttts_environment, -numpy.infty, variance_square=0, silent=True)

    for time in tqdm(range(horizon)):

        max_mean = -numpy.infty

        for arm in ttts_environment.listArms():
            if len(arm.history) != 0:
                mean_val = numpy.sum(arm.history) / len(arm.history)
                var_val = 1.0 / (len(arm.history) + 1)
            else:
                mean_val = 0
                var_val = 1.0

            sample_val = numpy.random.normal(mean_val, var_val)

            if sample_val >= max_mean:
                max_mean = sample_val
                max_mean_arm = arm

        if numpy.random.uniform() < sampling:
            sampled_arm = max_mean_arm
        else:
            max_mean_r = -numpy.infty
            max_mean_arm_r = max_mean_arm

            while max_mean_arm_r is max_mean_arm:
                for arm_r in ttts_environment.listArms():
                    if len(arm_r.history) != 0:
                        mean_val = numpy.sum(arm_r.history) / len(arm_r.history)
                        var_val = 1.0 / (len(arm_r.history) + 1)
                    else:
                        mean_val = 0
                        var_val = 1.0

                    sample_val_r = numpy.random.normal(mean_val, var_val)

                    if sample_val_r >= max_mean_r:
                        max_mean_r = sample_val_r
                        max_mean_arm_r = arm_r

            sampled_arm = max_mean_arm_r

        sampled_value = sampled_arm.pullArm()
        sampled_arm.environment.armPulled = numpy.append(sampled_arm.environment.armPulled, sampled_arm)
        sampled_arm.environment.gain = numpy.append(sampled_arm.environment.gain, sampled_value)
        sampled_arm.history = numpy.append(sampled_arm.history, sampled_value)
        ttts_environment.regret = numpy.append(ttts_environment.regret, optimal_mean - sampled_value)
        print("Time: " + str(time) + " Regret: " + str(ttts_environment.regret[time]))

    return ttts_environment

# Bayesian Elimination with Gaussian Arms

