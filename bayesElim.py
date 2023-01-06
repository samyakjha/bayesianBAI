import numpy as np
import scipy
from StochasticBanditsModules2 import GaussianArm
import math


class BanditInstance:
    def __init__(self, banditlist):
        self.banditlist = banditlist
        self.K = len(banditlist)
        self.varlist = bandit_vars(banditlist)
        self.activelist = np.ones(len(banditlist))
        self.meanlist = [bandit.mean for bandit in self.banditlist]
    
    def deact(self, index):
        self.activelist[index] = 0
        self.banditlist[index].deactivate()
        return 
        



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
    R = math.ceil(math.log(K)/math.log(2))
    round_budget = math.floor(n/R)
    #print('Budget', round_budget)
    #numSamp_list = np.floor((n/R)*bandits.varlist/np.sum(bandits.varlist)).astype(int)
    
    #print(numSamp_list)
    postmean_list = np.zeros(K)
    """
    for i in range(K):
        sumSamp = banditlist[i].pullArm_sum(numSamp_list[i])
        postMean = banditlist[i].posterior_mean(numSamp_list[i], sumSamp)
        postmean_list[i] = postMean
    red = math.ceil(K/2)
    tophalf_indices = np.argpartition(postmean_list, red)
    reduced_bandits = []
    for i in range(red):
        reduced_bandits.append(banditlist[tophalf_indices[i]])
    if K<=2:
        return reduced_bandits[0]
    elif K>=3:
        bayesElim(BanditInstance(reduced_bandits), n-math.floor(n/R))
    """
    for round in range(R):
        numSamp_list = round_budget*bandits.varlist*bandits.activelist/np.sum(bandits.varlist*bandits.activelist)
        numSamp_list = np.floor(numSamp_list).astype(int)
        #print('Sampling' , numSamp_list)
        num_act = np.sum(bandits.activelist)
        for i in range(K):
            
            sumSamp = banditlist[i].pullArm_sum(numSamp_list[i])
            postMean = banditlist[i].posterior_mean(numSamp_list[i], sumSamp)
            postmean_list[i] = postMean
        red = math.ceil(num_act/2)
        #print('r', red)
        #print('pml', postmean_list)
        tophalf_indices = np.argpartition(postmean_list, K-red)
        #print('thi', tophalf_indices)
        for j in range(0,K-red):
            bandits.deact(tophalf_indices[j])
        n = n - round_budget
        #print('actlist', bandits.activelist)
        
    
    #print(bandits.activelist)
    for i in range(K):
        if bandits.activelist[i]==1:
            #print (np.sum(bandits.activelist))
            return bandits.banditlist[i]
            break



            


        
        
        

"""
pm_list = np.array([1/2**i for i in range(1,9)])
v_list = np.array([0.25 for i in range(8)])
banditList = makeBandits(pm_list, v_list)
bandits = BanditInstance(banditList)
#print(bandits.K)
best = bayesElim(bandits, 50)
print(best.mean)
#print(np.floor((25/(np.log(3)/np.log(2))*bandits.varlist/np.sum(bandits.varlist))).astype(int))

#print(bandit_vars(makeBandits(pm_list, v_list)))
"""
"""
test = np.array([])
for i in range(3):
    np.append(test, [i])
for t in test:
    print(t)
"""    




