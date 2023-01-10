import numpy as np
import matplotlib.pyplot as plt
import StochasticBanditsModules2
from seq_halving import seqHalving
from bayesElim import BanditInstance, makeBandits, bandit_vars, bayesElim

N_runs = 15000

budget = [(50 + 50*i) for i in range(30)]

priori_means = np.array([1/2**i for i in range(8)])
var_list = np.array([0.5 for i in range(8)])
"""
banditlist_exp = bayesElim.makeBandits(priori_means, var_list)
bandits_exp = bayesElim.BanditInstance(banditlist_exp)
maxArm = max(bandits_exp.meanlist)
"""
num_mis_bayes = np.zeros(30)
num_mis_seq = np.zeros(30)
#print(maxArm)

for run in range(N_runs):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, var_list)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(30):
        if abs(bayesElim(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes[j] += 1
        if abs(seqHalving(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_seq[j] += 1
    
    
log_prob_mis_bayes = np.log((num_mis_bayes/N_runs))
log_prob_mis_seq = np.log((num_mis_seq/N_runs))
#print(log_prob_mis)
#print(num_mis)

plt.plot(budget, log_prob_mis_bayes )
plt.legend(['bayesElim'])
plt.plot(budget, log_prob_mis_seq)
plt.legend(['seqHalving'])
plt.xlabel('budget values')
plt.ylabel('log expected error probability')
plt.plot()
plt.savefig('plot_comb.png', dpi=300, bbox_inches='tight')
plt.show()


