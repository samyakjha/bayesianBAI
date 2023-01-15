import numpy as np
import matplotlib.pyplot as plt
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits
from alum import ALUM
from tqdm import tqdm

priori_means = [1/2**i for i in range(1,9)]
vars = [0.5 for i in range(len(priori_means))]
N_runs = 5000
Bandits = BanditInstance(makeBandits(priori_means, vars))
budget = [(25 + 25*i) for i in range(8)]

num_mis_alum = np.zeros(8)
#num_mis_bayes2 = np.zeros(10)
#print(maxArm)

for run in tqdm(range(N_runs)):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, vars)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(8):
        if abs(ALUM(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_alum[j] += 1
        """
        if abs(bayesElim2(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes2[j] += 1
        """
    
log_prob_mis_alum = np.log((num_mis_alum/N_runs))
#log_prob_mis_bayes2 = np.log((num_mis_bayes2/N_runs))
print(log_prob_mis_alum)
#print(num_mis)

#print(bandit.get_mean())

