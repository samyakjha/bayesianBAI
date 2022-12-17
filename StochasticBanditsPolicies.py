import StochasticBanditsModules
import numpy


def thompson_sampling_gaussian(arms, horizon):
    ts_environment = StochasticBanditsModules.Environment()

    for arm in arms:
        new_arm = StochasticBanditsModules.GaussianArm(ts_environment, arm.mean, arm.variance)

    mean_arr = numpy.array([])

    for arm in ts_environment.listArms():
        mean_arr = numpy.append(mean_arr, arm.mean)

    try:
        optimal_mean = numpy.amax(mean_arr)
    except ValueError:
        optimal_mean = -numpy.infty

    print("optimal_mean is " + str(optimal_mean))

    max_mean_arm = StochasticBanditsModules.GaussianArm(ts_environment, -numpy.infty, variance=0, silent=True)

    for time in range(horizon):
        max_mean = -numpy.infty

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
        try:
            max_mean_arm.environment.armPulled = numpy.append(max_mean_arm.environment.armPulled, max_mean_arm)
            max_mean_arm.environment.gain = numpy.append(max_mean_arm.environment.gain, max_mean_arm.pullArm(False))
            max_mean_arm.history = numpy.append(max_mean_arm.history, max_mean_arm.pullArm(False))
        except AttributeError:
            print("Max mean arm is None")

        ts_environment.regret = numpy.append(ts_environment.regret, optimal_mean - max_mean)
    return ts_environment


def ttts_gaussian(arms, horizon):
    ttts_environment = StochasticBanditsModules.Environment()

