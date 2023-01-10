import numpy as np
import matplotlib.pyplot as plt
import StochasticBanditsModules2
from seq_halving import seqHalving
from bayesElim import BanditInstance, makeBandits, bandit_vars

N_runs = 15000

budget = [(50 + 50*i) for i in range(30)]

priori_means = np.array([1/2**i for i in range(8)])
var_list = np.array([0.5 for i in range(8)])
"""
banditlist_exp = bayesElim.makeBandits(priori_means, var_list)
bandits_exp = bayesElim.BanditInstance(banditlist_exp)
maxArm = max(bandits_exp.meanlist)
"""
num_mis = np.zeros(30)
#print(maxArm)

for run in range(N_runs):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, var_list)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(30):
        if abs(seqHalving(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis[j] += 1
    
    
log_prob_mis = np.log((num_mis/N_runs))
print(log_prob_mis)
#print(num_mis)

plt.plot(budget, log_prob_mis )
plt.xlabel('budget values')
plt.ylabel('log expected error probability')
plt.title('seq halving budget vs error')
plt.savefig('log_plot_seq_halv.png', dpi=300, bbox_inches='tight')
plt.show()
