import numpy as np
import matplotlib.pyplot as plt
import StochasticBanditsModules2
#from seq_halving import seqHalving
from bayesElim import BanditInstance, makeBandits, bandit_vars, bayesElim
from bayesElim2 import bayesElim2
#from tqdm import tqdm

N_runs = 50000

budget = [(50 + 50*i) for i in range(100)]

priori_means = np.array([1/2**i for i in range(8)])
var_list = np.array([0.5 for i in range(8)])
"""
banditlist_exp = bayesElim.makeBandits(priori_means, var_list)
bandits_exp = bayesElim.BanditInstance(banditlist_exp)
maxArm = max(bandits_exp.meanlist)
"""
num_mis_bayes = np.zeros(100)
num_mis_bayes2 = np.zeros(100)
#print(maxArm)

for run in range(N_runs):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, var_list)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(100):
        if abs(bayesElim(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes[j] += 1
        if abs(bayesElim2(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes2[j] += 1
    
    
log_prob_mis_bayes = np.log((num_mis_bayes/N_runs))
log_prob_mis_bayes2 = np.log((num_mis_bayes2/N_runs))
#print(log_prob_mis)
#print(num_mis)

plt.plot(budget, log_prob_mis_bayes , color = 'red', label = 'bayesElim')
plt.plot(budget, log_prob_mis_bayes2, color = 'blue', label = 'bayesElim2')
plt.xlabel('budget values')
plt.ylabel('log expected error probability')
plt.legend()
plt.savefig('plot_comb_bayes_test.png', dpi=300, bbox_inches='tight')
plt.show()


