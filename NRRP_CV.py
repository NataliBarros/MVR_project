import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import pandas as pd
import AnalyzeTrace as at
plt.style.use('seaborn-deep')

# path to the raw data
RAW_DATA_PATH = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/invitro_raw.h5'#Project_MVR/MVR_warmupProject/h5_data/invitro_raw.h5' #invitro_raw_00.h5
RAW_DATA_PATH2 = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/noise_simulation_new_Noise_95/'
#csvfile_path = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/connsummaries/E2/connsummary_L5_TPCA-L5_TPCB.csv'

# Write computed values in file
#f = open('/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/TESTING-PROCEDURE/CV_profiles/SilicovsVitro_noise02_noJKK/CV_profiles.txt', 'w')

# total time of the recording
SIMULATION_TIME = 1.3
fs = 10000.0
fs2 = 1.0/0.000025

# times where a stimulus is performed
STIM_TIMES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 10000]
STIM_TIMES_silico = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 40000]
stim_times_s = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 1.0]

# lists used in plots
STIM_NUM = [1,2,3,4,5,6,7,8,9]
time_silico = np.arange(0,1.3,0.000025)
time_vitro = np.arange(0,1.3,0.0001)
t_wind_aft = 300
t_wind_bef = 50

# # dictionary with tau_mem values for this specific data
TAU_MEM_EACH = {'c0':20.9, 'c1':19.1, 'c2':30.4, 'c4':32.9, 'c10':33.3, 'c12':35.5, 'c13':43.5, 'c14':41.2, 'c31':31.7, 'c59':28.6, 'c60':46.9, 'c61':31.6, 'c62':23.0, 'c63':22.7, 'c64':34.1, 'c65':18.4, 'c66':29.1, 'c67':22.0, 'c68':37.1, 'c69':53.9, 'c70':18.7, 'c71':37.5, 'c73':36.7, 'c74':25.0, 'c75':39.0, 'c76':26.2, 'c77':48.0, 'c78':43.8, 'c79':17.1, 'c80':25.1, 'c81':25.6, 'c82':62.3, 'c83':46.7}
#TAU_MEM_EACH = {'c0':20.9}


''' ############### CV PROFILES IN VITRO VS IN SILICO - JKKestimator = mean ########################################## '''
# load raw invitro data
raw_data = h5py.File(RAW_DATA_PATH)

CV_vitro = []
# # we make a loop over all the connections in raw_data
list_key = TAU_MEM_EACH.keys()

for i in range(len(list_key)):
    sample_connection = np.transpose(raw_data[list_key[i]].value)
    sample_connection = sample_connection
    #print 'len sample_conn', len(sample_connection)

    amp_means = []
    for sweep_num in range(len(sample_connection)):
        # remove one different sweep from the trace set in each iteration and compute the mean
        new_sample = np.delete(sample_connection, sweep_num, 0)

        new_sample_mean = np.mean(new_sample, axis = 0)

        max1, min1, amplitudes1 = at.compute_amplitude2(new_sample_mean, STIM_TIMES, t_wind_aft)

        amp_means.append(amplitudes1)

    amp_MEAN1 = np.mean(amp_means, axis=0)

    #print 'amp1', amp_MEAN1

    DIF1 = []
    for am in amp_means:
        # print 'amp', am
        dif1 = (len(sample_connection - 1) * (am - amp_MEAN1)) ** 2
        # print 'dif2', dif2
        DIF1.append(dif1)

    N1 = np.float(len(sample_connection))
    #print 'sum1', np.sum(DIF1, axis=0)
    amp_std1 = np.sqrt(np.sum(DIF1, axis=0) / N1)
    #print 'std1', amp_std1

    CV1 = amp_std1 / amp_MEAN1
    CV_vitro.append(CV1)

CV_vitro_mean = np.mean(CV_vitro, axis=0)
print 'CV_vitro_JKK', CV_vitro_mean

#DIF = []
#for cv in CV_vitro_mean:
#    dif = (cv - CV_vitro)**2
#    DIF.append(dif)

CV_vitro_std = np.std(CV_vitro, axis = 0)
CV_vitro_stderr = CV_vitro_std/np.sqrt(len(CV_vitro))

print 'CV vitro std', CV_vitro_std
print 'CV vitro stderr', CV_vitro_stderr

######### IN SILICO EXPERIMENTS

#EPSP_amp2_dic = {}
CV_silico_dic = {}
NRRPdic = {}

lambda_values = np.arange(0.1, 12.9, 0.1)

for l in lambda_values:
    print 'computing CV for lambda = %.1f' %l
    NRRPdic['%.1f' %l]=[]
    nrrp = poisson.rvs(l, size=100, loc=1)
    #print nrrp
    nrrp_mean = np.mean(nrrp)
    nrrp_std = np.std(nrrp)
    NRRPdic['%.1f' % l].append(nrrp_mean)
    NRRPdic['%.1f' % l].append(nrrp_std)
    CV_silico_arr = []
    #EPSP2_arr = []
    for a, n in zip(range(1, 100), nrrp):
        try:
            file = 'noise_simulation_new02_%s.h5' % a
            raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
            if n > 24:
                sample_connection2 = raw_data2['nrrp24'].value
            else:
                sample_connection2 = raw_data2['nrrp%s' % n].value

            amp_means2 = []
            for sweep_num in range(len(sample_connection2)):

                new_sample2 = np.delete(sample_connection2, sweep_num, 0)

                new_sample_mean2 = np.mean(new_sample2, axis=0)

                max2, min2, amplitudes2 = at.compute_amplitude2(new_sample_mean2, STIM_TIMES_silico, 1200)

                amp_means2.append(amplitudes2)

            amp_MEAN2 = np.mean(amp_means2, axis=0)

            DIF2 = []
            for am in amp_means2:
                # print 'amp', am
                dif2 = (len(sample_connection2 - 1) * (am - amp_MEAN2)) ** 2
                # print 'dif2', dif2
                DIF2.append(dif2)

            N2 = np.float(len(sample_connection2))
            # print 'sum2', np.sum(DIF2, axis=0)
            amp_std2 = np.sqrt(np.sum(DIF2, axis=0) / N2)
            # print 'std2', amp_std2

            CV2 = amp_std2 / amp_MEAN2
            CV_silico_arr.append(CV2)

            #print CV_silicoJKK

        except KeyError:
            pass

    CV_silico_dic['%.1f' %l] = CV_silico_arr

# # #  WRITE CV PROFILES IN FILE AND COMPUTE DISTANCE
DISTANCE=[]
#f.write('IN SILICO \n')
for l in lambda_values:
    CV_silico = np.mean(CV_silico_dic['%.1f' % l], axis=0)
    #print CV_silico_dic['%.1f' % l]
    #print 'mean', CV_silico
    CV_silico_std = np.std(CV_silico_dic['%.1f' %l], axis=0)
    CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_silico_dic['%.1f' %l]))

    #f.write('CV profile = %s \n' %CV_silico)
    #f.write('CV std = %s \n' %CV_silico_std)
    #f.write('CV stderr = %s \n' %CV_silico_stderr)

    distance = []
    for i in range(9):
        d = ((CV_silico[i] - CV_vitro[i])/CV_vitro_stderr[i])**2
        distance.append(d)

    D = np.sqrt(np.abs(np.sum(distance)))
    DISTANCE.append(D)
    #f.write('distance to in vitro = %s \n' %D)

#f.close()

min_dis = np.min(DISTANCE)
for d, l in zip(DISTANCE, lambda_values):
    if d == min_dis:
        print 'min distance = %s for lambda = %s' %(d, l)
    else:
        pass

# plt.figure(1)
# plt.xlabel('NRRP')
# plt.ylabel('error')
# for d, l in zip(DISTANCE, lambda_values):
#     plt.plot(NRRPdic['%.1f' % l], d, 'b.')
# plt.show()

# ## PLOT AND SAVE CV PROFILES SILICO VS VITRO
# fig, ax = plt.subplots(figsize=(7, 4))
# plt.title('CV profile Vitro vs all Silico')
# plt.xlabel('# stimulus')
# plt.ylabel('CV value')
# ax.errorbar(STIM_NUM, CV_vitro, label='inVitro', color='red', yerr=CV_vitro_stderr, marker='.',capsize=5)

#plt.figure()
for l in lambda_values:
    CV_silico = np.mean(CV_silico_dic['%.1f' %l], axis=0)
    # print CV_silico[0]
    # plt.plot(NRRPdic['%.1f' % l], CV_silico[0], 'b.')
    CV_silico_std = np.std(CV_silico_dic['%.1f' %l], axis=0)
    CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_silico_dic['%.1f' %l]))

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.title('lambda=%.1f; nrrp=%.2f$\pm$%.2f' % (l, NRRPdic['%.1f' %l][0], NRRPdic['%.1f' %l][1]))
    plt.xlabel('# stimulus')
    plt.ylabel('CV value')
    ax.errorbar(STIM_NUM, CV_silico, label='inSilico', color = 'blue', yerr=CV_vitro_stderr, marker='o', linestyle='dotted', capsize=5)
    ax.errorbar(STIM_NUM, CV_vitro_mean, label='inVitro', color = 'red', yerr=CV_vitro_stderr, marker='o', linestyle='dotted', capsize=5)
    plt.legend()
    #plt.show()
    plt.savefig('/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/TESTING-PROCEDURE/CV_profiles/SilicovsVitro_new_noise02_newJKK/lambda_%.1f.png' %l)
#plt.show()


''' ############### CV PROFILES IN VITRO VS IN SILICO ########################################## '''
# # load raw invitro data
# raw_data = h5py.File(RAW_DATA_PATH)
#
# CV_connection = []
# CV_arr2 = []
# EPSP_vitro = []
# # # we make a loop over all the connections in raw_data
# list_key = TAU_MEM_EACH.keys()
#
# for i in range(len(list_key)):
#     sample_connection = np.transpose(raw_data[list_key[i]].value)
#     sample_connection = sample_connection
#
#     amp_means = []
#     amp_stds = []
#     for sweep_num in range(len(sample_connection)):
#         # remove one different sweep from the trace set in each iteration and compute the mean
#         new_sample = np.delete(sample_connection, sweep_num, 0)
#         new_samp_mean = np.mean(new_sample, axis = 0)
#         amp = []
#         for t in STIM_TIMES:
#             max = np.max(new_samp_mean[t:t + 300])
#             min = np.min(new_samp_mean[t:t + 300])
#             amp.append(np.abs(max - min))
#         amp_means.append(amp)
#
#         std_amp2 = []
#         for sweep in new_sample:
#             amp2 = []
#             for t in STIM_TIMES:
#                 max = np.max(sweep[t:t + 300])
#                 min = np.min(sweep[t:t + 300])
#                 amp2.append(np.abs(max - min))
#             std_amp2.append(amp2)
#         STDamp2 = np.std(std_amp2, axis=0)
#
#     mean = np.mean(amp_means, axis=0)
#     std = np.mean(STDamp2, axis = 0)
#
#     CV = std/mean
#     print CV
#     CV_connection.append(CV)
#
#
# # #print CV_arr1
# CV_vitro = np.mean(CV_connection, axis=0)
#
# DIF = []
# for cv in CV_connection:
#     dif = (cv - CV_vitro)**2
#     DIF.append(dif)
#
# CV_vitro_std = np.sqrt((len(CV_connection)-1)*np.sum(DIF))
# CV_vitro_stderr = CV_vitro_std/np.sqrt(len(CV_connection))
#
# plt.figure()
# plt.plot(CV_vitro)
# plt.show()
#f.write('IN VITRO \n')
#f.write('CV profile = %s \n' %CV_vitro)
#f.write('CV std = %s \n' %CV_vitro_std)
#f.write('CV stderr = %s \n' %CV_vitro_stderr)


## IN SILICO EXPERIMENTS

# EPSP_amp2_dic = {}
# CV_arr2_dic = {}
# NRRPdic = {}
#
# lambda_values = np.arange(0.1, 12.9, 0.1)
#
# for l in lambda_values:
#     print 'computing CV for lambda = %.1f' %l
#     NRRPdic['%.1f' %l]=[]
#     nrrp = poisson.rvs(l, size=100, loc=1)
#     #print nrrp
#     nrrp_mean = np.mean(nrrp)
#     nrrp_std = np.std(nrrp)
#     NRRPdic['%.1f' % l].append(nrrp_mean)
#     #NRRPdic['%.1f' % l].append(nrrp_std)
#     CV2_arr = []
#     EPSP2_arr = []
#     for a, n in zip(range(1, 100), nrrp):
#         try:
#             file = 'noise_simulation_new03_%s.h5' % a
#             raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
#             if n > 24:
#                 sample_connection2 = raw_data2['nrrp24'].value
#             else:
#                 sample_connection2 = raw_data2['nrrp%s' % n].value
#
#                 #CV2 = []
#                 AMP = []
#                 for trace in sample_connection2:
#                     amp = []
#                     for t in STIM_TIMES_silico:
#                         max = np.max(trace[t:t + 800])
#                         min = np.min(trace[t:t + 800])
#                         amp.append(np.abs(max - min))
#                     AMP.append(amp)
#
#                 # print 'amp', amp
#                 amp_mean = np.mean(AMP, axis=0)
#                 # print 'amp', amp_mean
#                 amp_std = np.std(AMP, axis=0)
#                 cv = amp_std / amp_mean
#                 #print 'cv', cv
#                 CV2_arr.append(cv)
#
#                 # mean_array2, max_array2, min_array2, maxtime_array2, amp_array2, CV2 = at.JKK_traces_GOOD(sample_connection2,
#                 #                                                                                     STIM_TIMES_silico,
#                 #                                                                                       800, fs2, 1.3)
#
#             #EPSP_amp2 = np.mean(amp_array2, axis=0)
#             #print 'CV', CV2
#             #EPSP2_arr.append(EPSP_amp2)
#             #print CV2_arr
#             #CV2_arr.append(CV2)
#
#             #print CV2_arr
#         except KeyError:
#             pass
#
#     CV_arr2_dic['%.1f' %l] = CV2_arr
#
#
# # #  WRITE CV PROFILES IN FILE AND COMPUTE DISTANCE
# DISTANCE=[]
# #f.write('IN SILICO \n')
# for l in lambda_values:
#     CV_silico = np.mean(CV_arr2_dic['%.1f' % l], axis=0)
#     #print 'mean', CV_silico
#     CV_silico_std = np.std(CV_arr2_dic['%.1f' %l], axis=0)
#     CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_arr2_dic['%.1f' %l]))
#
#     #f.write('CV profile = %s \n' %CV_silico)
#     #f.write('CV std = %s \n' %CV_silico_std)
#     #f.write('CV stderr = %s \n' %CV_silico_stderr)
#
#     distance = []
#     for i in range(9):
#         d = ((CV_silico[i] - CV_vitro[i])/CV_vitro_stderr[i])**2
#         distance.append(d)
#
#     D = np.sqrt(np.abs(np.sum(distance)))
#     DISTANCE.append(D)
#     #f.write('distance to in vitro = %s \n' %D)
#
# #f.close()
#
# min_dis = np.min(DISTANCE)
# for d, l in zip(DISTANCE, lambda_values):
#     if d == min_dis:
#         print 'min distance = %s for lambda = %s' %(d, l)
#     else:
#         pass
#
# # plt.figure(1)
# # plt.xlabel('NRRP')
# # plt.ylabel('error')
# # for d, l in zip(DISTANCE, lambda_values):
# #     plt.plot(NRRPdic['%.1f' % l], d, 'b.')
# # plt.show()
#
# # ## PLOT AND SAVE CV PROFILES SILICO VS VITRO
# # fig, ax = plt.subplots(figsize=(7, 4))
# # plt.title('CV profile Vitro vs all Silico')
# # plt.xlabel('# stimulus')
# # plt.ylabel('CV value')
# # ax.errorbar(STIM_NUM, CV_vitro, label='inVitro', color='red', yerr=CV_vitro_stderr, marker='.',capsize=5)
#
# plt.figure()
# for l in lambda_values:
#     CV_silico = np.mean(CV_arr2_dic['%.1f' %l], axis=0)
#     # print CV_silico[0]
#     # plt.plot(NRRPdic['%.1f' % l], CV_silico[0], 'b.')
#     CV_silico_std = np.std(CV_arr2_dic['%.1f' %l], axis=0)
#     CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_arr2_dic['%.1f' %l]))
#
#     fig, ax = plt.subplots(figsize=(7, 4))
#     #plt.title('lambda=%.1f; nrrp=%.2f$\pm$%.2f' % (l, NRRPdic['%.1f' %l][0], NRRPdic['%.1f' %l][1]))
#     plt.xlabel('# stimulus')
#     plt.ylabel('CV value')
#     ax.errorbar(STIM_NUM, CV_silico, label='inSilico', color = 'blue', yerr=CV_vitro_stderr, marker='o', linestyle='dotted', capsize=5)
#     ax.errorbar(STIM_NUM, CV_vitro, label='inVitro', color = 'red', yerr=CV_vitro_stderr, marker='o', linestyle='dotted', capsize=5)
#     plt.legend()
#     plt.show()
#     #plt.savefig('/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/TESTING-PROCEDURE/CV_profiles/SilicovsVitro_new_noise03_noJKK/lambda_%.1f.png' %l)
# plt.show()
##############################################################################################################

''' COMPUTE CV PROFILE FROM CONNSUMMARY FILE '''
# raw_data = h5py.File(RAW_DATA_PATH)
#
# CV_connection = []
# CV_arr2 = []
# EPSP_vitro = []
# # # we make a loop over all the connections in raw_data
# list_key = TAU_MEM_EACH.keys()
#
# for i in range(len(list_key)):
#     sample_connection = np.transpose(raw_data[list_key[i]].value)
#     sample_connection = sample_connection
#
#     AMP_sweep = []
#     for sweep in sample_connection:
#         amp = []
#         for t in STIM_TIMES:
#             max = np.max(sweep[t:t+200])
#             min = np.min(sweep[t:t+200])
#             amp.append(np.abs(max-min))
#         AMP_sweep.append(amp)
#
#     #print 'amp', AMP
#     amp_mean = np.mean(AMP_sweep, axis=0)*1000
#     #print 'amp_mean', amp_mean
#     amp_std = np.std(AMP_sweep, axis=0)*1000
#     cv = amp_std/amp_mean
#     #print 'cv', cv
#
#     CV_connection.append(cv)
#
# CV_vitro = np.mean(CV_connection, axis=0)
# CV_vitro_std = np.std(CV_connection, axis=0)
# CV_vitro_stderr = CV_vitro_std/np.sqrt(len(CV_connection))
#
# ######## in silico ########
# # define lambda values
# lambda_values = np.arange(0.0, 2.0, 0.1)
#
# data = pd.read_csv(csvfile_path)
#
# # find num of connections in file
# grouped = data.groupby(['pregid', 'postgid'])
# nconn = len(grouped)
#
# #EPSP_mean_dic = {}
# EPSP_CV_dic = {}
# NRRP_dic = {}
# for l in lambda_values:
#     NRRP_dic['%.1f' % l] = []
#     #EPSP_mean_dic['%.1f' % l] = []
#     EPSP_CV_dic['%.1f' % l] = []
#
# for l in lambda_values:
#     print 'computing CV for lambda = %.1f' % l
#
#     # Define nrrp values as a poisson distribution
#     nrrp = poisson.rvs(l, size=nconn, loc=1)
#     #print 'nrrp', nrrp
#     nrrp_mean = np.mean(nrrp)
#     #nrrp_std = np.std(nrrp)
#     NRRP_dic['%.1f' %l].append(nrrp_mean)
#     #NRRP_dic['%.1f' %l].append(nrrp_std)
#     EPSP_mean = []
#     EPSP_CV = []
#     n = 0
#     for stim, nrrp_val, d in zip(data['StimID'],data['Nrrp'],range(len(data['Nrrp']))):
#         #print 'nrrp[n]', nrrp[n]
#         if stim == 0 and n < nconn and nrrp_val == nrrp[n]:
#             #print 'EPSP mean', data['EPSP mean'][d]
#             #print 'EPSP cv', data['EPSP CV'][d]
#             #EPSP_mean.append(data['EPSP mean'][d])
#             if np.isnan(data['EPSP CV'][d]) == True :
#                 print 'Found CV = NaN'
#                 data['EPSP CV'][d] = 0.0
#                 EPSP_CV.append(data['EPSP CV'][d])
#             else:
#                 EPSP_CV.append(data['EPSP CV'][d])
#                 n = n+1
#
#         #print 'EPSP_mean', EPSP_mean
#
#         #EPSP_mean_dic['%.1f' % l].append(np.mean(EPSP_mean))
#         EPSP_CV_dic['%.1f' % l].append(np.mean(EPSP_CV))
#
# print EPSP_CV_dic['1.0']


''' COMPUTE CV AND FIRST EPSP IN SILICO COMPARE WITH IN VITRO - JKK '''

# raw_data = h5py.File(RAW_DATA_PATH)
#
# # we make a loop over all the connections in raw_data
# list_key = TAU_MEM_EACH.keys()
# #print len(list_key)
# EPSP_vitro = []
# CV_vitro = []
# for i in range(len(list_key)):
#     sample_connection = np.transpose(raw_data[list_key[i]].value)
#     sample_connection = sample_connection
#
#     mean_array, max_array, min_array, maxtime_array, amp_array, CV = at.JKK_traces(sample_connection,
#                                                                                           STIM_TIMES,
#                                                                                         300, fs, 1.3)
#
#     EPSP_amp = np.mean(amp_array, axis=0)
#     EPSP_vitro.append(EPSP_amp)
#     CV_vitro.append(CV)
#
# CV_vitro_mean = np.mean(CV_vitro, axis = 0)
# print np.mean(CV_vitro, axis = 0)
#
# EPSP_arr_dic = {}
# CV_arr_dic = {}
# #NRRPdic = {}
#
# # plt.figure()
# # plt.title('CV vs lambda')
# # plt.xlabel('lambda')
# # plt.ylabel('CV')
# #
# for l in range(1):
#     lambda_values = np.arange(0.1, 12.9, 0.1)
#
#     for l in lambda_values:
#         print 'computing CV for lambda = %.1f' %l
#         #NRRPdic['%.1f' %l]=[]
#         EPSP_arr_dic['%.1f' % l]=[]
#         CV_arr_dic['%.1f' % l]=[]
#         # Define nrrp values as a poisson distribution
#         nrrp = poisson.rvs(l, size=100, loc=1)
#         #print nrrp
#         nrrp_mean = np.mean(nrrp)
#         nrrp_std = np.std(nrrp)
#         #NRRPdic['%.1f' %l].append(nrrp_mean)
#         #NRRPdic['%.1f' %l].append(nrrp_std)
#
#         EPSP_silico = []
#         CV_silico = []
#         for a, n in zip(range(1, 99), nrrp):
#             try:
#                 file = 'noise_simulation_new03_%s.h5' % a
#                 raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
#                 if n > 24:
#                     sample_connection2 = raw_data2['nrrp24'].value
#                 else:
#                     sample_connection2 = raw_data2['nrrp%s' % n].value
#
#                 mean_array, max_array, min_array, maxtime_array, amp_array, CV = at.JKK_traces(sample_connection2,
#                                                                                            STIM_TIMES_silico,
#                                                                                            1200, fs2, 1.3)
#                 epsp_mean = np.mean(amp_array, axis = 0)
#                 EPSP_silico.append(epsp_mean)
#                 #print CV[0]
#                 CV_silico.append(CV[0])
#             except KeyError:
#                 pass
#
#         EPSP_arr_dic['%.1f' % l].append(EPSP_silico)
#         CV_arr_dic['%.1f' % l].append(CV_silico)
#
#     EPSP = []
#     CV = []
#     epsp_vitro = []
#     epsp_vitro_up = []
#     epsp_vitro_down = []
#     cv_vitro = []
#     for l in lambda_values:
#         cv_vitro.append(CV_vitro_mean[0])
#         #EPSP_mean = np.mean(EPSP_arr_dic['%.1f' % l])
#         #EPSP.append(EPSP_mean)
#         CV_mean = np.mean(CV_arr_dic['%.1f' % l])
#         CV.append(CV_mean)
#
#     plt.plot(lambda_values, CV, '.')
#     plt.plot(lambda_values, cv_vitro, 'r--')
#
# plt.show()



''' COMPUTE CV AND FIRST EPSP IN SILICO COMPARE WITH IN VITRO AND MARKRAM - No JKK '''

# raw_data = h5py.File(RAW_DATA_PATH)
#
# time_vitro = range(13000)
# time_silico = range(52000)
# # we make a loop over all the connections in raw_data
# list_key = TAU_MEM_EACH.keys()
# #print len(list_key)
# EPSP_vitro = []
# CV_vitro = []
# for i in range(len(list_key)):
#     sample_connection = np.transpose(raw_data[list_key[i]].value)
#     sample_connection = sample_connection
#
#     amp_list = []
#     for trace in sample_connection:
#         baseline = np.mean(trace[STIM_TIMES[0] - 50:STIM_TIMES[0]])
#         max_value = np.max(trace[STIM_TIMES[0]:STIM_TIMES[0] + 200])
#         for v, i in zip(trace[STIM_TIMES[0]:STIM_TIMES[0] + 200],
#                         time_vitro[STIM_TIMES[0]:STIM_TIMES[0] + 200]):
#             if v == max_value:
#                 EPSP_time = np.arange(i-3, i+3)
#                 time2 = i
#                 EPSP = trace[i-3:i+3]
#                 EPSP_mean = np.mean(trace[i-3:i+3])
#
#         amp = np.abs(EPSP_mean - baseline)
#         amp_list.append(amp)
#
#     epsp_mean = np.mean(amp_list)
#     epsp_std = np.std(amp_list)
#     cv = epsp_std / epsp_mean
#
#     EPSP_vitro.append(epsp_mean)
#     CV_vitro.append(cv)
#
# print np.mean(CV_vitro)
# print np.std(CV_vitro)
#
# EPSP_arr_dic = {}
# CV_arr_dic = {}
# #NRRPdic = {}
#
# lambda_values = np.arange(0.1, 12.9, 0.1)
#
# plt.figure()
# plt.title('CV vs lambda')
# plt.xlabel('lambda')
# plt.ylabel('CV')
#
#
# for rep in range(1):
#     print 'LOOP %s' %rep
#     for l in lambda_values:
#         print 'computing CV for lambda = %.1f' %l
#         #NRRPdic['%.1f' %l]=[]
#         EPSP_arr_dic['%.1f' % l]=[]
#         CV_arr_dic['%.1f' % l]=[]
#         # Define nrrp values as a poisson distribution
#         nrrp = poisson.rvs(l, size=100, loc=1)
#         #print nrrp
#         nrrp_mean = np.mean(nrrp)
#         nrrp_std = np.std(nrrp)
#         #NRRPdic['%.1f' %l].append(nrrp_mean)
#         #NRRPdic['%.1f' %l].append(nrrp_std)
#
#         EPSP_silico = []
#         CV = []
#         for a, n in zip(range(1, 99), nrrp):
#             try:
#                 file = 'noise_simulation_new03_%s.h5' % a
#                 raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
#                 if n > 24:
#                     sample_connection2 = raw_data2['nrrp24'].value
#                 else:
#                     sample_connection2 = raw_data2['nrrp%s' % n].value
#
#                 amp_arr = []
#                 for trace in sample_connection2:
#                     baseline = np.mean(trace[STIM_TIMES_silico[0] - 200:STIM_TIMES_silico[0]])
#                     max_value = np.max(trace[STIM_TIMES_silico[0]:STIM_TIMES_silico[0]+1200])
#                     for v, i in zip(trace[STIM_TIMES_silico[0]:STIM_TIMES_silico[0] + 1200],
#                                     time_silico[STIM_TIMES_silico[0]:STIM_TIMES_silico[0] + 1200]):
#                         if v == max_value:
#                             EPSP_time = np.arange(i-3, i+3)
#                             time2 = i
#                             EPSP = trace[i-3:i+3]
#                             EPSP_mean = np.mean(trace[i-3:i+3])
#                             #print 'EPSP', EPSP
#                             #print EPSP_mean
#                             #print baseline
#                     amp = np.abs(EPSP_mean - baseline)
#                     amp_arr.append(amp)
#                 epsp_mean = np.mean(amp_arr)
#                 # print 'epsp_mean', epsp_mean
#                 epsp_std = np.std(amp_arr)
#                 cv = epsp_std / epsp_mean
#                 # print 'cv', cv
#             except:
#                 pass
#
#             EPSP_silico.append(epsp_mean)
#             CV.append(cv)
#
#         EPSP_arr_dic['%.1f' % l].append(EPSP)
#         CV_arr_dic['%.1f' % l].append(CV)
#
#
#     EPSP = []
#     CV = []
#     cv_mk = []
#     cv_mk_up = []
#     cv_mk_down = []
#     cv_vitro = []
#     for l in lambda_values:
#         cv_mk.append(0.52)
#         cv_mk_up.append(0.52 + 0.03149*1.96)
#         cv_mk_down.append(0.52 - 0.03149 * 1.96)
#         cv_vitro.append(0.31)#(np.mean(CV_vitro))
#         EPSP_mean = np.mean(EPSP_arr_dic['%.1f' % l])
#         EPSP.append(EPSP_mean)
#         CV_mean = np.mean(CV_arr_dic['%.1f' % l])
#         CV.append(CV_mean)
#
#     plt.plot(lambda_values, CV, '.')
#     #plt.plot(lambda_values, cv_mk, 'r--', label = 'Markram')
#     #plt.plot(lambda_values, cv_mk_up, 'k--')
#     #plt.plot(lambda_values, cv_mk_down, 'k--')
#     plt.plot(lambda_values, cv_vitro, 'r--', label = 'in Vitro')
# plt.legend()
# plt.show()



