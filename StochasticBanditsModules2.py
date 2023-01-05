import numpy as np
import math


# noinspection PyTypeChecker
class GaussianArm:
    def __init__(self, priori_mean,variance, priori_variance=0.5, silent=False):
        self.variance = variance
        self.mean = np.random.normal(priori_mean, math.sqrt(priori_variance))
        self.priori_mean = priori_mean
        self.priori_variance = priori_variance
        #self.environment = environment
        #self.history = numpy.array([])
        #self.silent = silent
        """
        if silent is False:
            environment.K = numpy.append(environment.K, self)
        else:
            pass
        """
    @classmethod
    def sum_val(cls, arm):
        return np.sum(arm.history)

    def __del__(self):
        pass

    def pullArm(self, num_samp):

        X = np.random.normal(self.mean, math.sqrt(self.variance), num_samp )
        return X
    def pullArm_sum(self, num_samp):
        X = np.random.normal(self.mean, math.sqrt(self.variance), num_samp )
        return np.sum(X)
    
    def posterior_variance(self, num_samp):
        varinv = 1/self.priori_variance + num_samp/self.variance
        return 1/varinv

    def posterior_mean(self, num_samps, sum_samps):
        n = num_samps
        pv = self.posterior_variance(n)
        return pv*(self.priori_mean/self.priori_variance + sum_samps/self.variance)
    def get_variance(self):
        return self.variance
"""
class PrioriGaussianArm(GaussianArm):

    def __init__(self, priori_mean, priori_variance_square=0.5, silent=False):
        self.priori_mean = priori_mean
        self.priori_variance_square = priori_variance_square

        mean = numpy.random.normal(priori_mean, priori_variance_square)

        self.posterior_mean = mean
        self.posterior_variance_square = 0.5

        super().__init__(mean, 0.5, silent)
"""



# TODO: BernoulliArm, BetaArm, ExponentialArm, GammaArm, PoissonArm, UniformArm
