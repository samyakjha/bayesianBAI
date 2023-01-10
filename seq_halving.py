import numpy as np
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits, bandit_vars


def seqHalving(bandits, n):
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
        numSamp_list = round_budget*bandits.activelist/np.sum(bandits.activelist)
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