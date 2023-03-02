import numpy as np
import matplotlib.pyplot as plt
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits, bayesElim
from alum import ALUM
from tqdm import tqdm
from experiment import unimodal_bandits_np
from seq_halving import seqHalving
from bayesALUM import bayesALUM
k_star = 12
K = 70
nu_star, nu_one, nu_K = 0.7, 0.1, 0.3
priori_means = [nu_one]
for ind in range(2, k_star):
    priori_means.append(nu_one + (ind-1)*(nu_star - nu_one)/(k_star-1))
priori_means.append(nu_star)
for ind in range(k_star+1, K):
    priori_means.append(nu_K - (ind-k_star)*(nu_star-nu_K)/(K-k_star-1))
priori_means.append(nu_K)


num_budget = 70
vars = [0.5 for i in range(len(priori_means))]
N_runs = 50000
Bandits = BanditInstance(makeBandits(priori_means, vars))
Bandits = unimodal_bandits_np(Bandits)
budget = [(250 + 250*i) for i in range(num_budget)]

num_mis_alum = np.zeros(num_budget)
num_mis_bayesalum = np.zeros(num_budget)
num_mis_bayesElim = np.zeros(num_budget)
num_mis_seqhalv = np.zeros(num_budget)
#num_mis_seq = np.zeros(num_budget)
#print(maxArm)

for run in tqdm(range(N_runs)):
    #priori_means = np.array([1/2**i for i in range(8)])
    #var_list = np.array([0.5 for i in range(8)])
    banditlist_exp = makeBandits(priori_means, vars)
    bandits_exp = BanditInstance(banditlist_exp)
    unimodal_bandits_np(bandits_exp)
    maxArm = max(bandits_exp.meanlist)
    #print('bandit means orig', [bandit.get_mean() for bandit in bandits_exp.banditlist])
    
    
    for j in range(num_budget):
        if abs(ALUM(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_alum[j] += 1
        if abs(bayesALUM(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayesalum[j] += 1
        if abs(bayesElim(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayesElim[j] += 1
        if abs(seqHalving(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_seqhalv[j] += 1
        
        """
        if abs(seqHalving(bandits_exp, budget[j]).get_mean() - maxArm)>10e-4:
            num_mis_seq[j] += 1
        """
        """
        if abs(bayesElim2(bandits_exp, budget[j]).get_mean()-maxArm)>10e-4:
            num_mis_bayes2[j] += 1
        """
    
prob_mis_alum = num_mis_alum/N_runs
prob_mis_bayesalum = num_mis_bayesalum/N_runs
prob_mis_bayesElim = num_mis_bayesElim/N_runs
prob_mis_seqhalv = num_mis_seqhalv/N_runs
#prob_mis_seq = num_mis_seq/N_runs
print('ALUM ',prob_mis_alum)
print('BayesALUM ', prob_mis_bayesalum)
print('Seq Halving', prob_mis_seqhalv)
print('Bayes Elim', prob_mis_bayesElim)
#log_prob_mis_bayes2 = np.log((num_mis_bayes2/N_runs))

#print(num_mis)

#print(bandit.get_mean())
plt.plot(budget, prob_mis_alum, color = 'cyan', label = 'ALUM')
plt.plot(budget, prob_mis_bayesalum, color = 'red', label = 'bayesALUM')
plt.plot(budget, prob_mis_bayesElim, color = 'green', label = 'bayesElim')
plt.plot(budget, prob_mis_seqhalv, color = 'blue', label = 'SeqHalving')
#plt.plot(budget, prob_mis_seq, color = 'green', label = 'seq halving')
plt.xlabel('budget values')
plt.ylabel('expected error probability')
plt.legend()
#plt.yscale('log')
plt.savefig('alum_and_bayesalum.png', dpi=300, bbox_inches='tight')
plt.show()
