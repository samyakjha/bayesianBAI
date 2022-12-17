import numpy


class Environment:
    def __init__(self):
        # K: array containing arms
        self.K = numpy.array([])
        # armPulled : array containing the arms pulled
        self.armPulled = numpy.array([])
        # gainTS : array containing the gain for each arm pulled
        self.gain = numpy.array([])
        # regret : array containing the regret for each arm pulled
        self.regret = numpy.array([])

    def listArms(self):
        return self.K


class GaussianArm:
    def __init__(self, environment, mean, variance=1, silent=False):
        self.mean = mean
        self.variance = variance
        self.environment = environment
        self.history = numpy.array([])
        self.silent = silent

        if silent is False:
            environment.K = numpy.append(environment.K, self)
        else:
            pass

    def __del__(self):
        pass

    def pullArm(self, log_val=True):

        gain_val = numpy.random.normal(self.mean, self.variance)

        if not self.silent and log_val:
            self.environment.armPulled = numpy.append(self.environment.armPulled, self)
            self.environment.gain = numpy.append(self.environment.gain, gain_val)
            self.history = numpy.append(self.history, gain_val)
            return None
        else:
            return gain_val

# TODO: BernoulliArm, BetaArm, ExponentialArm, GammaArm, PoissonArm, UniformArm
