import numpy
import pandas
import matplotlib.pyplot as plt


class Environment:
    def __init__(self):
        # K: array containing arms
        self.K = pandas.array([])
        # armPulled : array containing the arms pulled
        self.armPulled = pandas.array([])
        # gainTS : array containing the gain for each arm pulled
        self.gain = pandas.array([])

    def listArms(self):
        return self.K


class GaussianArm:
    def __init__(self, environment, mean, variance=1):
        self.mean = mean
        self.variance = variance
        self.environment = environment
        environment.K.append(self)

    def __del__(self):
        pass

    def pullArm(self):
        self.environment.armPulled.append(self)
        self.environment.gain.append(numpy.random.normal(self.mean, self.variance))

# TODO: BernoulliArm, BetaArm, ExponentialArm, GammaArm, PoissonArm, UniformArm
