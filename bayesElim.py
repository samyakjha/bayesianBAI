import numpy as np
import scipy
from StochasticBanditsModules2 import GaussianArm
import math


class BanditInstance:
    def __init__(self, banditlist):
        self.banditlist = banditlist
        self.K = len(banditlist)
        self.varlist = bandit_vars(banditlist)
        



def makeBandits(priori_mean_list, var_list):
    K = len(priori_mean_list)
    banditList = []
    for i in range(K):
        banditList.append(GaussianArm(priori_mean_list[i], var_list[i]))
    return banditList
    

def bandit_vars(banditlist):
    K = len(banditlist)
    varlist = np.zeros(K)
    for i in range(K):
        varlist[i] = banditlist[i].get_variance()
    return varlist



def bayesElim(bandits, n):
    K = bandits.K
    banditlist = bandits.banditlist
    R = np.ceil(np.log(K)/np.log(2))
    numSamp_list = np.floor((n/R)*bandits.varlist/np.sum(bandits.varlist)).astype(int)
    print(bandits.varlist[0])
    postmean_list = np.zeros(K)
    for i in range(K):
        sumSamp = banditlist[i].pullArm_sum(numSamp_list[i])
        postMean = banditlist[i].posterior_mean(numSamp_list[i], sumSamp)
        postmean_list[i] = postMean
    red = math.ceil(K/2)
    tophalf_indices = np.argpartition(postmean_list, red)
    reduced_bandits = []
    for i in range(red+1):
        reduced_bandits.append(banditlist[tophalf_indices[i]])
    if K<=1:
        return banditlist[0]
    elif K>=2:
        bayesElim(BanditInstance(reduced_bandits), n-math.floor(n/R))


pm_list = np.array([1,2,3])
v_list = np.array([0.1,0.2,0.3])
banditList = makeBandits(pm_list, v_list)
bandits = BanditInstance(banditList)
#print(bandits.K)
bayesElim(bandits, 25)
#print(np.floor((25/(np.log(3)/np.log(2))*bandits.varlist/np.sum(bandits.varlist))).astype(int))

#print(bandit_vars(makeBandits(pm_list, v_list)))
"""
test = np.array([])
for i in range(3):
    np.append(test, [i])
for t in test:
    print(t)
"""    




