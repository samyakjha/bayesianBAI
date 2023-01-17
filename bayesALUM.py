import numpy as np
import math
from StochasticBanditsModules2 import GaussianArm
from bayesElim import BanditInstance, makeBandits


def bayesALUM(bandits, budget):
    L = math.floor(math.log((bandits.K)/3)/math.log(3/2))
    #print(L
    numSamp = [2**(L-2)*budget//3**(L-1) for i in range(1,3)]
    for l in range(3,L+2):
        numSamp.append(2**(L-(l-1))*budget//3**(L-(l-2)))
    #print(numSamp)
    for l in range(1, L+1):
        sampledBandits = bandits.four_samples()
        postMean = np.array([])
        for bandit in sampledBandits:
            sum_samp = bandit.pullArm_sum(numSamp[l-1]//4)
            post_mean = bandit.posterior_mean(numSamp[l-1]//4, sum_samp)
            postMean = np.append(postMean, post_mean)
        #print('Sampled means at round ' ,sampledMean/(numSamp[l-1]//4))
        bestind = np.argmax(postMean)
        if bestind == 0 or bestind == 1:
            bandits.deact_in_series(2*bandits.numact()//3 + 1, bandits.numact())
        else:
            bandits.deact_in_series(0, math.ceil(bandits.numact()/3))
        
    active = bandits.active_indices()
    #print('Active at end', active)
    postMeans = np.array([bandits.banditlist[ind].posterior_mean(numSamp[-1]//3, bandits.banditlist[ind].pullArm_sum(numSamp[-1]//3)) for ind in active])
    #print('Sampled mean end', sampledMeans)
    maxInd = active[np.argmax(postMeans)]
    #print('max', maxInd)

    for j in range(bandits.K):
        bandits.reset(j)

    return bandits.banditlist[maxInd]