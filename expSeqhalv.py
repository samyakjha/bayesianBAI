import numpy as np
import matplotlib.pyplot as plt
import StochasticBanditsModules2
from seq_halving import seqHalvHistory, seqHalving
from bayesElim import BanditInstance, makeBandits, bandit_vars, bayesElim
from bayesElim2 import bayesElim2
from tqdm import tqdm

N_runs = 10000

budget = [(50 + 50*i) for i in range(15)]

priori_means = np.array([1/2**i for i in range(8)])
var_list = np.array([0.5 for i in range(8)])
"""
banditlist_exp = bayesElim.makeBandits(priori_means, var_list)
bandits_exp = bayesElim.BanditInstance(banditlist_exp)
maxArm = max(bandits_exp.meanlist)
"""
num_mis_seqhalvhist = np.zeros(15)
num_mis_seqhalv = np.zeros(15)
num_mis_bayes2 = np.zeros(15)
num_mis_bayes = np.zeros(15)
#print(maxArm)

for run in tqdm(range(N_runs)):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, var_list)
    bandits_exp = BanditInstance(banditlist_exp)
    maxArm = max(bandits_exp.meanlist)
    
    
    for j in range(15):
        if abs(seqHalvHistory(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_seqhalvhist[j] += 1
        if abs(bayesElim2(bandits_exp, budget[j]).get_mean() - maxArm)>10e-4:
            num_mis_bayes2[j] += 1
        if abs(seqHalving(bandits_exp, budget[j]).get_mean() - maxArm)>10e-4:
            num_mis_seqhalv[j] += 1
        if abs(bayesElim(bandits_exp, budget[j]).get_mean() - maxArm)>10e-4:
            num_mis_bayes[j] += 1
    
    
prob_mis_seqhist = num_mis_seqhalvhist/N_runs
prob_mis_bayes2 = num_mis_bayes2/N_runs
prob_mis_seq = num_mis_seqhalv/N_runs
prob_mis_bayes = num_mis_bayes/N_runs
#print(prob_mis_seq)
#print(prob_mis_bayes)
#print(num_mis)

plt.plot(budget, prob_mis_bayes2 , color = 'red', label = 'bayesElim2')
plt.plot(budget, prob_mis_seqhist, color = 'blue', label = 'seqHalvHist')
plt.plot(budget, prob_mis_seq, color = 'green', label = 'seqHalv')
plt.plot(budget, prob_mis_bayes, color = 'cyan', label = 'bayesElim')
plt.xlabel('budget values')
plt.ylabel('expected error probability')
plt.legend()
plt.yscale('log')
plt.savefig('seqHalv_vs_bayesElim.png', dpi=300, bbox_inches='tight')
plt.show()
