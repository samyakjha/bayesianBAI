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


# noinspection PyTypeChecker
class GaussianArm:
    def __init__(self, environment, mean, variance_square=0.5, silent=False):
        self.mean = mean
        self.variance_square = variance_square
        self.environment = environment
        self.history = numpy.array([])
        self.silent = silent

        if silent is False:
            environment.K = numpy.append(environment.K, self)
        else:
            pass

    @classmethod
    def sum_val(cls, arm):
        return numpy.sum(arm.history)

    def __del__(self):
        pass

    def pullArm(self, log_val=True):

        gain_val = numpy.random.normal(self.mean, numpy.sqrt(self.variance_square))

        if not self.silent and log_val:
            self.environment.armPulled = numpy.append(self.environment.armPulled, self)
            self.environment.gain = numpy.append(self.environment.gain, gain_val)
            self.history = numpy.append(self.history, gain_val)


        else:
            pass

        return gain_val
    def sum_val(self):
        return numpy.sum(self.history)

class PrioriGaussianArm(GaussianArm):

    def __init__(self, environment, priori_mean, priori_variance_square=0.5, silent=False):
        self.priori_mean = priori_mean
        self.priori_variance_square = priori_variance_square

        mean = numpy.random.normal(priori_mean, priori_variance_square)

        self.posterior_mean = mean
        self.posterior_variance_square = 0.5

        super().__init__(environment, mean, 0.5, silent)


def arm_mean(arm_p):
    return arm_p.mean

def arm_empirical_mean(arm_p):
    if len(arm_p.history) != 0:
        return numpy.sum(arm_p.history) / len(arm_p.history)
    else:
        return int(0)
def arm_posterior_mean(arm_p):
    return arm_p.posterior_mean


def arm_priori_mean(arm_p):
    return arm_p.priori_mean


def arm_variance_square(arm_p):
    return arm_p.variance_square


def arm_posterior_variance_square(arm_p):
    return arm_p.posterior_variance_square


def arm_priori_variance_square(arm_p):
    return arm_p.priori_variance_square

# TODO: BernoulliArm, BetaArm, ExponentialArm, GammaArm, PoissonArm, UniformArm
