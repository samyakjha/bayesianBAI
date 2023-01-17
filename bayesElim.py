import numpy as np
import scipy
from StochasticBanditsModules2 import GaussianArm
import math


class BanditInstance:
    def __init__(self, banditlist):
        self.banditlist = banditlist
        self.K = len(banditlist)
        self.varlist = bandit_vars(banditlist)
        self.activelist = np.ones(len(banditlist)).astype(int)
        self.meanlist = np.array([self.banditlist[i].get_mean() for i in range(self.K)])
        self.priori_meanlist = np.array([self.banditlist[i].get_priori_mean() for i in range(self.K)])
        
    
    def deact(self, index):
        self.activelist[index] = 0
        self.banditlist[index].deactivate()
        return 
    
    def reset(self, index):
        self.activelist[index] = 1
        self.banditlist[index].activate()
        return

    def active_indices(self):
        indlist = []
        for i in range(self.K):
            if self.activelist[i]==1:
                indlist.append(i)
        return indlist

    def four_samples(self):
        act_ind = self.active_indices()
        print('Active', self.active_indices())
        numAct = len(act_ind)
        indices = [act_ind[0], act_ind[math.ceil(numAct/3)-1], act_ind[2*numAct//3-1], act_ind[numAct-1]]
        print('Sampled', indices)
        sampList = [self.banditlist[ind] for ind in indices]
        
        return sampList
    
    def deact_in_series(self, start, end):
        act = self.active_indices()
        for ind in range(start, end):
            self.deact(act[ind])
        return 
    
    def numact(self):
        return np.sum(self.activelist)

    def rearrange(self, new_indices):
        old_banditlist = self.banditlist
        new_banditlist = []
        for ind in new_indices:
            new_banditlist.append(old_banditlist[ind])
        self.banditlist = new_banditlist
        self.varlist = bandit_vars(self.banditlist)
        self.activelist = np.ones(len(self.banditlist)).astype(int)
        self.meanlist = np.array([self.banditlist[i].get_mean() for i in range(self.K)])
        self.priori_meanlist = np.array([self.banditlist[i].get_priori_mean() for i in range(self.K)])
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

def bandit_means(banditlist):
    K = len(banditlist)
    meanlist = np.zeros(K)
    for i in range(K):
        meanlist[i] = banditlist[i].get_mean()
    return meanlist


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



            


        
        
        


pm_list = np.array([1, 2, 3, 4])
v_list = np.array([0.25 for i in range(8)])
banditList = makeBandits(pm_list, v_list)
bandits = BanditInstance(banditList)
#print(bandits.active_indices())
#best = bayesElim(bandits, 25)
#print(best.mean)
#print(np.floor((25/(np.log(3)/np.log(2))*bandits.varlist/np.sum(bandits.varlist))).astype(int))

#print(bandit_vars(makeBandits(pm_list, v_list)))

"""
test = np.array([])
for i in range(3):
    np.append(test, [i])
for t in test:
    print(t)
"""    




