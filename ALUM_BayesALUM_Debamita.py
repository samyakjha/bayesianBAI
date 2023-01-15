#importing the libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
#from is_natural_number import isNaturalNumber


#Defining the functions
def argmax_index(b):
  if mean_rewards[b] > mean_rewards[0]:
    return b
  else:
   return 0
  
def argmin_index(b):
  if mean_rewards[b] < mean_rewards[-1]:
    return b
  else:
   return 0

def samples_l(mean_rewards,T):
    K = len(mean_rewards)
    L = int((np.log2(K/3))/(np.log2(3/2)))
    No_l = np.zeros(L+1)
    for l in range(L+1):
        if l==0:
            No_l[l] = ((2**(L-2))/(3**(L-1)))*T
        elif l==1:
            No_l[l] = ((2**(L-2))/(3**(L-1)))*T
        else:
            No_l[l] = ((2**(L-(l-1)))/(3**(L-(l-2))))*T
            
    return No_l
            

def ALUM(T, mean_rewards, sigma):
  K = len(mean_rewards)
  L = int((np.log2(K/3))/(np.log2(3/2)))
  #Define N_l
  N_l = samples_l(mean_rewards,T)
  T_r = [0 for i in range(K)]
  mu = [0 for i in range(K)]
  B = []
  B.append([i for i in range(K)])
  for l in range(L):
    #print('------------------------------------')
    #print(B[l])
    #print('------------------------------------')
    K_M = B[l][0]
    K_N = B[l][-1]
    K_A = B[l][int(len(B[l])/3)]
    K_B = B[l][int(2*len(B[l])/3)]
    S_sorted = [K_M, K_A, K_B, K_N]
    #S_sorted = [i for n, i in enumerate(S) if i not in S[:n]]
    for i in range(len(S_sorted)):
      sample = np.random.normal(mean_rewards[S_sorted[i]], sigma[S_sorted[i]], int(N_l[l]//len(S_sorted)))
      T_r[S_sorted[i]] = T_r[S_sorted[i]] + N_l[l]//len(S_sorted)
      mu[S_sorted[i]] = ((T_r[S_sorted[i]] - N_l[l]//len(S_sorted))*mu[S_sorted[i]] + sum(sample))/T_r[S_sorted[i]]
    mu_s = [mu[K_M], mu[K_A], mu[K_B], mu[K_N]]
    x_l = S_sorted[np.argmax(mu_s)]
    if x_l == K_M or x_l == K_A:
      B.append([i for i in range(K_M, K_B+1)])
    elif x_l == K_B or x_l == K_N:
      B.append([i for i in range(K_B, K_N+1)])
    
    #if len(B[l]) < 1:
    #    #print('\n',x_l, (T_k - T)*mean_rewards[x_l])
    #    return x_l     #, (T_k - T)*mean_rewards[x_l]
    #break
  #print(B[L-1])
  #print('-----------------------')
  
  for i in range(len(B[L-1])):
      sample_L = np.random.normal(mean_rewards[B[L-1][i]], sigma[B[L-1][i]], int(N_l[L]//len(B[L-1])))
      T_r[B[L-1][i]] = T_r[B[L-1][i]] + N_l[l]//len(B[L-1])
      mu[B[L-1][i]] = ((T_r[B[L-1][i]] - N_l[l]//len(B[L-1]))*mu[B[L-1][i]] + sum(sample_L))/T_r[B[L-1][i]]
  mu_L = [mu[i] for i in B[L-1]]
  x_L = B[L-1][np.argmax(mu_L)]

  return x_L           #B[L-1], x_l, (T_k - T)*mean_rewards[x_l]

def Bayes_ALUM(T, mean_rewards, nu, sigma, sigma0):
  K = len(mean_rewards)
  L = int((np.log2(K/3))/(np.log2(3/2)))
  #Define N_l
  N_l = samples_l(mean_rewards,T)
  #post_var = [0 for i in range(K)]
  post_mu = [0 for i in range(K)]
  B = []
  B.append([i for i in range(K)])
  for l in range(L):
    #print('------------------------------------')
    #print(B[l])
    #print('------------------------------------')
    K_M = B[l][0]
    K_N = B[l][-1]
    K_A = B[l][int(len(B[l])/3)]
    K_B = B[l][int(2*len(B[l])/3)]
    S_sorted = [K_M, K_A, K_B, K_N]
    #S_sorted = [i for n, i in enumerate(S) if i not in S[:n]]
    for i in range(len(S_sorted)):
      sample = np.random.normal(mean_rewards[S_sorted[i]], sigma[S_sorted[i]], int(N_l[l]//len(S_sorted)))
      post_var = 1/((1/(sigma0**2)) + (N_l[l]/(sigma[S_sorted[i]]**2)))                                          #T_r[S_sorted[i]] + N_l[l]//len(S_sorted)
      post_mu[S_sorted[i]] = post_var*((nu[S_sorted[i]]/(sigma0**2)) + (sum(sample)/(sigma[S_sorted[i]]**2)))    #(T_r[S_sorted[i]] - N_l[l]//len(S_sorted))*mu[S_sorted[i]] + sum(sample)/T_r[S_sorted[i]]
    post_mu_s = [post_mu[K_M], post_mu[K_A], post_mu[K_B], post_mu[K_N]]
    post_x_l = S_sorted[np.argmax(post_mu_s)]
    if post_x_l == K_M or post_x_l == K_A:
      B.append([i for i in range(K_M, K_B+1)])
    elif post_x_l == K_B or post_x_l == K_N:
      B.append([i for i in range(K_B, K_N+1)])
    
  
  for i in range(len(B[L-1])):
      sample_L = np.random.normal(mean_rewards[B[L-1][i]], sigma[B[L-1][i]], int(N_l[L]//len(B[L-1])))
      post_var = 1/((1/(sigma0**2)) + (N_l[l]/(sigma[B[L-1][i]]**2)))                                             #T_r[B[L-1][i]] + N_l[l]//len(B[L-1])
      post_mu[B[L-1][i]] = post_var*((nu[B[L-1][i]]/(sigma0**2)) + (sum(sample_L)/(sigma[B[L-1][i]]**2)))         # (T_r[B[L-1][i]] - N_l[l]//len(B[L-1]))*mu[B[L-1][i]] + sum(sample_L)/T_r[B[L-1][i]]
  post_mu_L = [post_mu[i] for i in B[L-1]]
  post_x_L = B[L-1][np.argmax(post_mu_L)]

  return post_x_L           #B[L-1], x_l, (T_k - T)*mean_rewards[x_l]

# =============================================================================
# #Defining the required function
# def adding_inf_h_j(Q, h, j):
#   while len(Q)<h+1:
#     Q.append([0])
#   while len(Q[h])<j+1:
#     Q[h].append(np.inf)
#   return Q
# 
# def adding_0_h_j(Q, h, j):
#   while len(Q)<h+1:
#     Q.append([0])
#   while len(Q[h])<j+1:
#     Q[h].append(0)
#   return Q
# 
# #Defining the main function which breaks using a stopping criterion
# def hba(mean_rewards, T, ita, gamma, rho, sgma_sq):
#   K = len(mean_rewards)
#   tau = [[0, 1]]
#   Q = [[np.inf, np.inf], [np.inf, np.inf, np.inf]]
#   N = [[0, 0], [0,0,0]]
#   R = [[0, 0], [0,0,0]]
#   E = [[np.inf, np.inf], [np.inf, np.inf, np.inf]]
#   x_l = 0
#   x_h = 1
#   C =[]
#   t=0
#   while 1<2:
#     t+=1
#     h = 0
#     j = 1
#     P = [[h, j]]
#     while 1<2:
#       x_a = x_l + (x_h - x_l)/2
#       if [h,j] in tau:
#         if Q[h+1][2*j -1] > Q[h+1][2*j]:
#           #print(h+1, 2*j-1)
#           h +=1
#           j = 2*j -1
#           x_h = x_a
#         elif Q[h+1][2*j -1] < Q[h+1][2*j]:
#           #print(h+1, 2*j-1)
#           h +=1
#           j = 2*j
#           x_l = x_a
#         else:
#           if np.random.binomial(1, 0.5)==1:
#             #print(h+1, 2*j-1)
#             h +=1
#             j = 2*j -1
#             x_h = x_a
#           else:
#             h +=1
#             j = 2*j
#             x_l = x_a
#         P.append([h,j])
#         N = adding_0_h_j(N, h, j)
#         R = adding_0_h_j(R, h, j)
#       else:
#         N = adding_0_h_j(N, h, j)
#         R = adding_0_h_j(R, h, j)
#         h_t = h
#         j_t = j
#         c_t = (x_l+x_h)/2
#         C.append(c_t)
#         break
#     tau.append([h_t, j_t])
#     Q = adding_inf_h_j(Q, h_t, j_t)
#     Q = adding_inf_h_j(Q, h_t+1, 2*j_t-1)
#     Q = adding_inf_h_j(Q, h_t+1, 2*j_t+1)
#     E = adding_inf_h_j(E, h_t, j_t)
#     #Atttribute Update
#     arm = int(c_t*K) + 1
#     r_t = np.random.binomial(1, mean_rewards[arm-1])
#     for i in P:
#       N[i[0]][i[1]] = N[i[0]][i[1]]+1
#       R[i[0]][i[1]] = ((N[i[0]][i[1]]-1)*R[i[0]][i[1]] + r_t)/N[i[0]][i[1]]
#     for i in tau:  
#       if N[i[0]][i[1]]>0:
#         E[i[0]][i[1]] = R[i[0]][i[1]] + np.sqrt((2*sgma_sq*np.log(t))/N[i[0]][i[1]]) + rho*(gamma**i[0])
#         Q[i[0]][i[1]] = min(E[i[0]][i[1]], max(Q[i[0]+1][2*i[1] -1 ], Q[i[0]+1][2*i[1]]))
# 
#     #Stopping Criterian
#     if x_h - x_l < ita/K:
#       #print('------------------------------------------------------')
#       #print('The best arm is: ', arm)
#       #print('------------------------------------------------------')
#       #print('Time: ', t)
#       #print('------------------------------------------------------')
#       #print('Mean Reward:', mean_rewards[arm-1])
#       #print('------------------------------------------------------')
#       throughput = (T-t)*mean_rewards[arm-1]
#       #print('Throughput: ', throughput)
#       break
# 
#   return arm, t, mean_rewards[arm-1], throughput
# =============================================================================

#Importing the file
# =============================================================================
# means_file = 'unimodal.txt' 
# mean_rewards = np.loadtxt(means_file).tolist()
# max_arm = np.argmax(mean_rewards) + 1
# =============================================================================

#################### GENERATE UNIMODAL MEANS ##############
def unimodal_mean(best_arm, mean, nu):
    mean_sorted = np.sort(mean)
    mean1 = mean_sorted[:best_arm]
    mean2 = np.flip(mean_sorted[best_arm:])
    new_mean = np.concatenate((mean1, mean2))
    new_nu = [0]*len(mean_sorted)
    for i in range(len(new_mean)):
        for j in range(len(mean)):
            if new_mean[i] == mean[j]:
                new_nu[i] = nu[j]

    return new_mean, new_nu

K = 10
a = (1/2)
nu_values = [0]*K
for i in range(K):
    nu_values[i] = a**i   #np.array([1, a, a**2, a**3, a**4, a**5, a**6, a**7, a**8, a**9])
sigma0 = 0.5
sigma = [0.5]*len(nu_values)
K = len(nu_values)
mean_values = np.zeros(K)
for i in range(len(nu_values)):
    mean_values[i] = np.random.normal(nu_values[i], sigma0)

max_arm = int(K/2)+1
mean_rewards, nu = unimodal_mean(max_arm, mean_values, nu_values)

np.savetxt("mean_rewards_ALUM_BayesALUM_SeqHav_BayesElim.txt", mean_rewards)
np.savetxt("nu_ALUM_BayesALUM_SeqHav_BayesElim.txt", nu)

#print("Mean rewards:", mean_rewards)
#print("Mean values:", mean_values)
#print("Nu values:", nu_values)
#print("Nu:", nu)
#plt.plot(np.arange(K), mean_rewards)


runs = 1000
T1 = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
tinv = 1.9623

prob_err_ALUM = [0]*len(T1)
uconf_prob_err_ALUM = [0]*len(T1)
lconf_prob_err_ALUM = [0]*len(T1)

prob_err_BayesALUM = [0]*len(T1)
uconf_prob_err_BayesALUM = [0]*len(T1)
lconf_prob_err_BayesALUM = [0]*len(T1)
for i in tqdm(range(len(T1))):
    err_ALUM = np.zeros(runs)
    err_BayesALUM = np.zeros(runs)
    for r in range(runs):
        x_l_ALUM = ALUM(T1[i], mean_rewards, sigma)
        x_l_BayesALUM = Bayes_ALUM(T1[i], mean_rewards, nu, sigma, sigma0)
        if x_l_ALUM==max_arm:
            err_ALUM[r] = 0
        else:
            err_ALUM[r] = 1
        prob_err_ALUM[i] = np.sum(err_ALUM)/runs
        uconf_prob_err_ALUM[i] = prob_err_ALUM[i] + ((tinv*np.std(err_ALUM))/np.sqrt(runs-1))
        lconf_prob_err_ALUM[i] = prob_err_ALUM[i] - ((tinv*np.std(err_ALUM))/np.sqrt(runs-1))
        
        if x_l_BayesALUM==max_arm:
            err_BayesALUM[r] = 0
        else:
            err_BayesALUM[r] = 1
        prob_err_BayesALUM[i] = np.sum(err_BayesALUM)/runs
        uconf_prob_err_BayesALUM[i] = prob_err_BayesALUM[i] + ((tinv*np.std(err_BayesALUM))/np.sqrt(runs-1))
        lconf_prob_err_BayesALUM[i] = prob_err_BayesALUM[i] - ((tinv*np.std(err_BayesALUM))/np.sqrt(runs-1))

    
print(prob_err_ALUM)
print(prob_err_BayesALUM)

np.savetxt("prob_err_ALUM_K_" + str(K) + "_" +".txt" , prob_err_ALUM)
np.savetxt("uconf_prob_err_ALUM_K_" + str(K) + "_" +".txt" , uconf_prob_err_ALUM)
np.savetxt("lconf_prob_err_ALUM_K_" + str(K) + "_" +".txt" , lconf_prob_err_ALUM)

np.savetxt("prob_err_BayesALUM_K_" + str(K) + "_" +".txt" , prob_err_BayesALUM)
np.savetxt("uconf_prob_err_BayesALUM_K_" + str(K) + "_" +".txt" , uconf_prob_err_BayesALUM)
np.savetxt("lconf_prob_err_BayesALUM_K_" + str(K) + "_" +".txt" , lconf_prob_err_BayesALUM)



# =============================================================================
# prob_err_seqHav      = np.array([1.,    0.9,    0.8,    0.312, 0.306, 0.248, 0.224, 0.192, 0.172, 0.144]) #, 0.124, 0.112])
# uconf_prob_err_seqHav= np.array([1.,    1.,    1.,    0.35269927, 0.34648141, 0.28593586, 0.24272184, 0.22659963, 0.20515084, 0.19484134]) #, 0.17652671, 0.13970324])
# lconf_prob_err_seqHav= np.array([1.,   1.,   1.,   0.27130073, 0.26551859, 0.21006414, 0.20527816, 0.15740037, 0.13884916, 0.12315866]) # 0.11147329, 0.08429676])
#  
# prob_err_BayesElim = np.array([0.834, 0.434,  0.335,  0.146, 0.124, 0.118, 0.11, 0.106, 0.09, 0.08]) #,  0.07, 0.06])
# uconf_prob_err_BayesElim = np.array([1,  0.4498517,  0.3998517, 0.17295196, 0.16748568, 0.15295196, 0.14701848, 0.13304188, 0.11633939, 0.09586021]) #, 0.08748568, 0.07241329])
# lconf_prob_err_BayesElim = np.array([1.,   0.31498152, 0.29504804, 0.19504804, 0.14966061, 0.12251432,0.09251432, 0.08895812, 0.07541398, 0.04758671]) #, 0.02501483, 0.01501483])
# 
# 
# prob_err_ALUM= [0.115, 0.02, 0.006, 0.001, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0]
# prob_err_BayesALUM = [0.508, 0.295, 0.189, 0.12, 0.095, 0.056, 0.042, 0.02, 0.028, 0.02]
# 
# uconf_prob_err_BayesALUM = [0.539038238593587, 0.32331315237456143, 0.21330658739726355, 0.14017504995941435, 0.1132040806953197, 0.07027455802753421, 0.0544534608407202, 0.02869181943030393, 0.0382422366913229, 0.02869181943030393]
# 
# =============================================================================




markers = ['*', 'o', 'v', '+']
labels1 = 'ALUM'
labels2 = 'Bayes_ALUM'
labels4 = 'BayesElim'
labels3 = 'Seq. Halv.'
line_type = ['-', '-.', '--', ':']
plt.rcParams['pdf.fonttype'] = 42

for lower, upper, T in zip(lconf_prob_err_ALUM, uconf_prob_err_ALUM, T1):
    plt.plot((T,T), (upper, lower), line_type[0], color = 'r', alpha = 0.4)
    plt.plot((T - 0.125, T + 0.125), (upper, upper), line_type[0], color = 'r', alpha = 0.4)
    plt.plot((T - 0.125, T + 0.125), (lower, lower), line_type[0], color = 'r', alpha = 0.4)

plt.plot(T1, prob_err_ALUM, 
          line_type[0], 
          color = 'r', 
          marker = markers[0], 
          label = labels1)

for lower, upper, T in zip(lconf_prob_err_BayesALUM, uconf_prob_err_BayesALUM, T1):
    plt.plot((T,T), (upper, lower), line_type[1], color = 'b', alpha = 0.4)
    plt.plot((T - 0.125, T + 0.125), (upper, upper), line_type[1], color = 'b', alpha = 0.4)
    plt.plot((T - 0.125, T + 0.125), (lower, lower), line_type[1], color = 'b', alpha = 0.4)

plt.plot(T1, prob_err_BayesALUM, 
          line_type[1], 
          color = 'b', 
          marker = markers[1], 
          label = labels2)


# =============================================================================
# for lower, upper, T in zip(lconf_prob_err_seqHav, uconf_prob_err_seqHav, T1):
#     plt.plot((T,T), (upper, lower), line_type[0], color = 'g', alpha = 0.4)
#     plt.plot((T - 0.125, T + 0.125), (upper, upper), line_type[0], color = 'g', alpha = 0.4)
#     plt.plot((T - 0.125, T + 0.125), (lower, lower), line_type[0], color = 'g', alpha = 0.4)
# 
# plt.plot(T1, prob_err_seqHav, 
#           line_type[0], 
#           color = 'g', 
#           marker = markers[0], 
#           label = labels3)
# 
# for lower, upper, T in zip(lconf_prob_err_BayesElim, uconf_prob_err_BayesElim, T1):
#     plt.plot((T,T), (upper, lower),line_type[1], color = 'k', alpha = 0.4)
#     plt.plot((T - 0.125, T + 0.125), (upper, upper), line_type[1], color = 'k', alpha = 0.4)
#     plt.plot((T - 0.125, T + 0.125), (lower, lower), line_type[1], color = 'k', alpha = 0.4)
# 
# plt.plot(T1, prob_err_BayesElim, 
#       line_type[1], 
#       color = 'k', 
#       marker = markers[1], 
#       label = labels4)
# =============================================================================
    
legend_properties = {'weight':'bold'}
plt.legend(loc='upper right', fontsize = 10, prop=legend_properties)#, fontweight='bold')
plt.xlabel('Budget (T)', fontsize=14, fontweight='bold')
plt.ylabel('Error Probability', fontsize=14, fontweight='bold')
#plt.yscale('log')

plt.grid()
plt.savefig('BayesALUM_ALUM_ConfInterval_Proberr_unimodal_plot_trial.png')




