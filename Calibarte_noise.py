import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import ou_tester

RAW_DATA_PATH_vitro = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/invitro_raw.h5'
RAW_DATA_PATH_silico = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/noise_simulation_new_Noise_400/'

TAU_MEM_EACH = {'c0':20.9, 'c1':19.1, 'c2':30.4, 'c4':32.9, 'c10':33.3, 'c12':35.5, 'c13':43.5, 'c14':41.2, 'c31':31.7, 'c59':28.6, 'c60':46.9, 'c61':31.6, 'c62':23.0, 'c63':22.7, 'c64':34.1, 'c65':18.4, 'c66':29.1, 'c67':22.0, 'c68':37.1, 'c69':53.9, 'c70':18.7, 'c71':37.5, 'c73':36.7, 'c74':25.0, 'c75':39.0, 'c76':26.2, 'c77':48.0, 'c78':43.8, 'c79':17.1, 'c80':25.1, 'c81':25.6, 'c82':62.3, 'c83':46.7}

# load raw invitro data
raw_data = h5py.File(RAW_DATA_PATH_vitro)

# we make a loop over all the connections in raw_data
list_key = TAU_MEM_EACH.keys()
#list_key = ['c14']
#print len(list_key)

TAU = []
V_MEAN = []
SIGMA = []

for a in range(1,31):
    try:
        file = 'noise_simulation_new03_%s.h5' %a
        raw_data2 = h5py.File(RAW_DATA_PATH_silico+file)
        print file
        tau_nrrp = []
        v_mean_nrrp = []
        sigma_nrrp = []
        for n in range(1, 25, 1):
            NRRP = 'nrrp%d' %n
            sample_connection = raw_data2[NRRP].value

            tau, v_corr, v_mean, v_std = ou_tester.ou_test_silico_NAT(sample_connection, p_initial=[5], upper_only=False)

            tau_nrrp.append(np.mean(tau))
            v_mean_nrrp.append(np.mean(v_mean))
            sigma_nrrp.append(np.mean(v_std))

        TAU.append(np.mean(tau_nrrp))
        V_MEAN.append(np.mean(v_mean_nrrp))
        SIGMA.append(np.mean(sigma_nrrp))
    except KeyError:
        pass

print 'tau', np.mean(TAU)
print 'v_mean', np.mean(V_MEAN)
print 'v_std', np.mean(SIGMA)

# TAU = []
# V_MEAN = []
# SIGMA = []
# for i in range(len(list_key)):
#     print list_key[i]
#     sample_connection = np.transpose(raw_data[list_key[i]].value)
#     sample_connection = sample_connection
#
#     #tau, v_corr = ou_tester.ou_test_vitro(sample_connection, p_initial=[5], upper_only=False)
#     #print 'Eilif', tau
#
#     tau_N, v_corr_N, v_mean_N, v_std_N = ou_tester.ou_test_vitro_NAT(sample_connection, p_initial=[5], upper_only=False)
#
#     TAU.append(np.mean(tau_N))
#     V_MEAN.append(np.mean(v_mean_N))
#     SIGMA.append(np.mean(v_std_N)) # plot the distribution of this
#
#
# print 'tau', np.mean(TAU)
# print 'v_mean', np.mean(V_MEAN)
# print 'v_std - sigma', np.mean(SIGMA)

# tau and sigma distributions
plt.figure(1)
plt.title('NEW_noise03 sigma = %.2f' %np.mean(SIGMA))
sns.distplot(SIGMA, bins=10, rug=False)

plt.figure(2)
plt.title('NEW_noise03 Tau = %.2f' %np.mean(TAU))
sns.distplot(TAU, color=".2", bins=10, rug=False)
plt.show()
