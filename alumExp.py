import numpy as np
import matplotlib.pyplot as plt
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits
from alum import ALUM
from tqdm import tqdm
from experiment import unimodal_bandits_np
k_star = 6
K = 20
nu_star, nu_one, nu_K = 0.7, 0.1, 0.3
priori_means = [nu_one]
for ind in range(2, k_star):
    priori_means.append(nu_one + (ind-1)*(nu_star - nu_one)/(k_star-1))
priori_means.append(nu_star)
for ind in range(k_star+1, K):
    priori_means.append(nu_K - (ind-k_star)*(nu_star-nu_K)/(K-k_star-1))
priori_means.append(nu_K)



vars = [0.5 for i in range(len(priori_means))]
N_runs = 15000
Bandits = BanditInstance(makeBandits(priori_means, vars))
Bandits = unimodal_bandits_np(Bandits)
budget = [(250 + 250*i) for i in range(50)]

num_mis_alum = np.zeros(50)
#print(maxArm)

for run in tqdm(range(N_runs)):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, vars)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(50):
        if abs(ALUM(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_alum[j] += 1
        """
        if abs(bayesElim2(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes2[j] += 1
        """
    
prob_mis_alum = num_mis_alum/N_runs
#log_prob_mis_bayes2 = np.log((num_mis_bayes2/N_runs))

#print(num_mis)

#print(bandit.get_mean())
plt.plot(budget, prob_mis_alum, color = 'cyan', label = 'ALUM')
plt.xlabel('budget values')
plt.ylabel('expected error probability')
plt.legend()
plt.yscale('log')
plt.savefig('ALUMplot.png', dpi=300, bbox_inches='tight')
plt.show()
