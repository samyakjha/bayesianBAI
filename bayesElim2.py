import numpy as np
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits, bandit_vars

def bayesElim2(bandits, n):
    K = bandits.K
    banditlist = bandits.banditlist
    R = math.ceil(math.log(K)/math.log(2))
    round_budget = math.floor(n/R)
    #print('Budget', round_budget)
    #numSamp_list = np.floor((n/R)*bandits.varlist/np.sum(bandits.varlist)).astype(int)
    
    #print(numSamp_list)
    postmean_list = np.zeros(K)
    totalSamp_list = np.zeros(K).astype(int)
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
        totalSamp_list += numSamp_list
        #print('Total Sampling', totalSamp_list)
        for i in range(K):
            
            Samp = banditlist[i].pullArm(numSamp_list[i])
            #print('Samples', Samp)
            banditlist[i].add_to_history(Samp)

            sumSamp = np.sum(banditlist[i].history)
            postMean = banditlist[i].posterior_mean(totalSamp_list[i], sumSamp)
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
        
    bestBand = bandits.banditlist[0]
    #print(bandits.activelist)
    for i in range(K):
        if bandits.activelist[i]==1:
            #print (np.sum(bandits.activelist))
            bestBand = bandits.banditlist[i]
            break

    for j in range(K):
        bandits.reset(j)

    return bestBand

"""
pm_list = np.array([1, 2, 3, 4])
v_list = np.array([0.1, 0.2, 0.3, 0.4])
banditList = makeBandits(pm_list, v_list)
bandits = BanditInstance(banditList)
#print(bandits.K)
best = bayesElim2(bandits, 25)
print('History', best.history)
#print(np.floor((25/(np.log(3)/np.log(2))*bandits.varlist/np.sum(bandits.varlist))).astype(int))

#print(bandit_vars(makeBandits(pm_list, v_list)))
"""