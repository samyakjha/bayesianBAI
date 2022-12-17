import numpy as np
import StochasticBanditsModules
import StochasticBanditsPolicies
import matplotlib

experiment = StochasticBanditsModules.Environment()
arm1 = StochasticBanditsModules.GaussianArm(experiment, 1)
arm2 = StochasticBanditsModules.GaussianArm(experiment, 0.5)
arm3 = StochasticBanditsModules.GaussianArm(experiment, 0.25)
arm4 = StochasticBanditsModules.GaussianArm(experiment, 0.125)
arm5 = StochasticBanditsModules.GaussianArm(experiment, 0.0625)
arm6 = StochasticBanditsModules.GaussianArm(experiment, 0.03125)
arm7 = StochasticBanditsModules.GaussianArm(experiment, 0.015625)
arm8 = StochasticBanditsModules.GaussianArm(experiment, 0.0078125)

tsg_environment = StochasticBanditsPolicies.thompson_sampling_gaussian(experiment.listArms(), 5000)
matplotlib.pyplot.plot(np.arange(5000), tsg_environment.regret)
matplotlib.pyplot.show()
