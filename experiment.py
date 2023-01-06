import numpy as np
import matplotlib.pyplot as plt
import StochasticBanditsModules2
import bayesElim

N_runs = 5000

horizons = [25, 50, 75, 100, 125, 150, 175, 200]
priori_means = np.array([1/2**i for i in range(8)])
var_list = np.array([0.5*0.5 for i in range(8)])
banditlist_exp = bayesElim.makeBandits(priori_means, var_list)
bandits_exp = bayesElim.BanditInstance(banditlist_exp)
maxArm = max(bandits_exp.meanlist)
num_mis = np.zeros(8)
print(maxArm)

for run in range(N_runs):
    bestBandits = [bayesElim.bayesElim(bandits_exp, horizon) for horizon in horizons]
    """
    for i in range(8):
        if abs(bestBandits[i].mean-maxArm)>10e-4:
            num_mis[i] = num_mis[i] + 1
    """
    print([bestBandits[i].mean for i in range(8)])
prob_mis = (num_mis/N_runs)
#print(prob_mis)
#print(num_mis)




