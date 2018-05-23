# Run AnalyzeTrace

import h5py
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from scipy.stats import poisson
import AnalyzeTrace as at
plt.style.use('seaborn-deep')

# path to the raw data
RAW_DATA_PATH = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/invitro_raw.h5'#Project_MVR/MVR_warmupProject/h5_data/invitro_raw.h5' #invitro_raw_00.h5
RAW_DATA_PATH2 = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/noise_simulation_new_OldNoise/'

#f = open('/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/TESTING-PROCEDURE/CV_profiles/SilicovsVitro_new_noise01/CV_profiles.txt', 'w')

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
#TAU_MEM_EACH = {'c10':33.3}

''' ############### CV PROFILES IN VITRO VS IN SILICO ########################################## '''
# load raw invitro data
raw_data = h5py.File(RAW_DATA_PATH)

first_EPSP_amp = []
first_EPSP_amp_2 = []
CV_arr1 = []
CV_arr2 = []
CV1_first = []
CV2_first = []

# we make a loop over all the connections in raw_data
list_key = TAU_MEM_EACH.keys()
#print len(list_key)
for i in range(len(list_key)):
    sample_connection = np.transpose(raw_data[list_key[i]].value)
    sample_connection = sample_connection*1000.0

    mean_array, max_array, min_array, maxtime_array, amp_array, CV = at.JKK_traces_GOOD(sample_connection, STIM_TIMES, t_wind_aft, fs, 1.3)

    #EPSP_amp = np.mean(amp_array, axis=0)
    #first_EPSP_amp.append(EPSP_amp[0]*1000.0)
    #print 'CV', CV
    CV_arr1.append(CV)
    #CV1_first.append(CV[0])

CV_vitro = np.mean(CV_arr1, axis=0)
CV_vitro_std = np.std(CV_arr1, axis=0)
CV_vitro_stderr = CV_vitro_std/np.sqrt(len(CV_arr1))

# #print CV_vitro
#plt.figure()
#plt.plot(CV_vitro)
#plt.show()

# f.write('IN VITRO \n')
# f.write('CV profile = %s \n' %CV_vitro)
# f.write('CV std = %s \n' %CV_vitro_std)
# f.write('CV stderr = %s \n' %CV_vitro_stderr)


##### IN SILICO EXPERIMENTS ---------------------------------------------------
EPSP_amp2_dic = {}
CV_arr2_dic = {}
NRRPdic = {}

lambda_values = np.arange(0.1, 2.9, 0.1)

for l in lambda_values:
    print 'computing CV for lambda = %.1f' % l
    NRRPdic['%.1f' % l] = []
    nrrp = poisson.rvs(l, size=100, loc=1.0)
    #print nrrp
    nrrp_mean = np.mean(nrrp)
    nrrp_std = np.std(nrrp)
    NRRPdic['%.1f' % l].append(nrrp_mean)
    NRRPdic['%.1f' % l].append(nrrp_std)
    CV2_arr = []
    EPSP2_arr = []
    for a, n in zip(range(1, 101), nrrp):
        file = 'noise_simulation_%s.h5' % a
        raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
        if n > 24:
            sample_connection2 = raw_data2['nrrp24'].value
        else:
            sample_connection2 = raw_data2['nrrp%s' % n].value

        mean_array2, max_array2, min_array2, maxtime_array2, amp_array2, CV2 = at.JKK_traces_GOOD(sample_connection2,
                                                                                                 STIM_TIMES_silico,
                                                                                                 1200, fs2, 1.3)
        EPSP_amp2 = np.mean(amp_array2, axis=0)
        CV2_arr.append(CV2)
        EPSP2_arr.append(EPSP_amp2)

    CV_arr2_dic['%.1f' %l] = CV2_arr

#  WRITE CV PROFILES IN FILE AND COMPUTE DISTANCE
DISTANCE=[]
#f.write('IN SILICO \n')
for l in lambda_values:
    CV_silico = np.mean(CV_arr2_dic['%.1f' % l], axis=0)
    CV_silico_std = np.std(CV_arr2_dic['%.1f' %l], axis=0)
    CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_arr2_dic['%.1f' %l]))

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

# ## PLOT AND SAVE CV PROFILES SILICO VS VITRO
for l in lambda_values:
    CV_silico = np.mean(CV_arr2_dic['%.1f' %l], axis=0)
    CV_silico_std = np.std(CV_arr2_dic['%.1f' %l], axis=0)
    CV_silico_stderr = CV_silico_std/np.sqrt(len(CV_arr2_dic['%.1f' %l]))

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.title('lambda=%.1f; nrrp=%.2f$\pm$%.2f' % (l, NRRPdic['%.1f' %l][0], NRRPdic['%.1f' %l][1]))
    plt.xlabel('# stimulus')
    plt.ylabel('CV value')
    ax.errorbar(STIM_NUM, CV_silico, label='inSilico', yerr=CV_silico_stderr, marker='o', linestyle='dotted', capsize=5)
    ax.errorbar(STIM_NUM, CV_vitro, label='inVitro', color = 'blue', yerr=CV_vitro_stderr, marker='o', linestyle='dotted', capsize=5)
    plt.legend()
    plt.show()
    #plt.savefig('/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/TESTING-PROCEDURE/CV_profiles/SilicovsVitro_new_noise02/lambda_%.1f.png' %l)


#raw_data = h5py.File(RAW_DATA_PATH)

#sample_connection = np.transpose(raw_data['c60'].value)
#sample_connection = sample_connection


# file = 'noise_simulation_new03_20.h5'
# raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
#
# sample_connection2 = raw_data2['nrrp24'].value
#
#
# sample_norm = []
# plt.figure()
# plt.title('L5_TTPC in Silico')
# plt.xlabel('time (s)')
# plt.ylabel('Voltage norm (mV)')
# for trace in sample_connection2:
#     trace_norm = (trace - trace[0])
#     sample_norm.append(trace_norm)
#     plt.plot(time_silico, trace_norm, 'b')#'k--', alpha=0.3, linewidth = 0.1)
#     plt.show()
# mean_norm = np.mean(sample_norm, axis=0)
# plt.plot(time_silico, mean_norm, 'b-', linewidth = 1)
# plt.ylim(-0.7,4)
# plt.show()

# raw_data = h5py.File(RAW_DATA_PATH)
#
# sample_connection = np.transpose(raw_data['c59'].value)
# sample_connection = sample_connection
# M = np.mean(sample_connection, axis=0)
# plt.figure()
# plt.plot(M, 'r')
# plt.show()
# for sweep in sample_connection:
#     plt.figure()
#     plt.plot(sweep, 'C2')
#     plt.show()


''' COMPUTE CV AND FIRST EPSP IN SILICO COMPARE WITH IN VITRO AND MARKRAM '''

# raw_data = h5py.File(RAW_DATA_PATH)
#
# # we make a loop over all the connections in raw_data
# list_key = TAU_MEM_EACH.keys()
# #print len(list_key)
# EPSP_vitro = []
# CV_vitro = []
# for i in range(len(list_key)):
#     sample_connection = np.transpose(raw_data['c1'].value)
#     sample_connection = sample_connection
#
#
#     amp_list = []
#     for trace in sample_connection:
#         amp = np.abs(np.max(trace[STIM_TIMES[0]:STIM_TIMES[0] + 300]) - np.min(
#                                     trace[STIM_TIMES[0]:STIM_TIMES[0] + 300]))
#         amp_list.append(amp)
#
#     epsp_mean = np.mean(amp_list)
#     epsp_std = np.std(amp_list)
#     cv = epsp_std / epsp_mean
#
#     EPSP_vitro.append(epsp_mean)
#     CV_vitro.append(cv)
#
# print CV_vitro
#
# EPSP_arr_dic = {}
# CV_arr_dic = {}
# #NRRPdic = {}
#
# lambda_values = np.arange(0.0, 12.9, 0.1)
#
# for l in lambda_values:
#     print 'computing CV for lambda = %.1f' %l
#     #NRRPdic['%.1f' %l]=[]
#     EPSP_arr_dic['%.1f' % l]=[]
#     CV_arr_dic['%.1f' % l]=[]
#     nrrp = poisson.rvs(l, size=100, loc=1)
#     #print nrrp
#     nrrp_mean = np.mean(nrrp)
#     nrrp_std = np.std(nrrp)
#     #NRRPdic['%.1f' %l].append(nrrp_mean)
#     #NRRPdic['%.1f' %l].append(nrrp_std)
#
#     EPSP = []
#     CV = []
#     for a, n in zip(range(1, 101), nrrp):
#         file = 'noise_simulation_new02_%s.h5' % a
#         raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
#         if n > 24:
#             sample_connection2 = raw_data2['nrrp24'].value
#         else:
#             sample_connection2 = raw_data2['nrrp%s' % n].value
#
#         amp_arr = []
#         for trace in sample_connection2:
#             #print np.max(trace[STIM_TIMES_silico[0]:STIM_TIMES_silico[0]+1200])
#             amp = np.abs(np.max(trace[STIM_TIMES_silico[0]:STIM_TIMES_silico[0]+1200]) - np.min(trace[STIM_TIMES_silico[0]:STIM_TIMES_silico[0]+1200]))
#             amp_arr.append(amp)
#
#         epsp_mean = np.mean(amp_arr)
#         epsp_std = np.std(amp_arr)
#         cv = epsp_std/epsp_mean
#
#         EPSP.append(epsp_mean)
#         CV.append(cv)
#
#     EPSP_arr_dic['%.1f' % l].append(EPSP)
#     CV_arr_dic['%.1f' % l].append(CV)
#
#
#
#
# EPSP = []
# CV = []
# epsp_vitro = []
# epsp_vitro_up = []
# epsp_vitro_down = []
# cv_mk = []
# for l in lambda_values:
#     cv_mk.append(np.mean(CV_vitro))
#     EPSP_mean = np.mean(EPSP_arr_dic['%.1f' % l])
#     EPSP.append(EPSP_mean)
#     CV_mean = np.mean(CV_arr_dic['%.1f' % l])
#     CV.append(CV_mean)
#
# plt.figure(1)
# plt.title('CV vs lambda')
# plt.xlabel('lambda')
# plt.ylabel('CV')
# plt.plot(lambda_values, CV, 'o')
# plt.plot(lambda_values, cv_mk, 'r--')
# plt.show()




##############################################################################################################


'''Ex. U_data DISTRIBUTION - NORMAL (Gaussian) '''
# #mean, var and std
# avg_ = np.mean(U_data)
# var_U = np.var(U_data)
# sigma_U = np.sqrt(var_U) #STD
# sem_U = sigma_U/np.sqrt(len(U_data))
#
# print '(U norm dist) -> Uloc=Umean = %s, Uscale=Ustd = %s, Usem = %s' %(avg_U, sigma_U, sem_U)
#
# #From these two values we know the shape of the fitted Gaussian
# pdf_x = np.linspace(np.min(U_data),np.max(U_data),80)
# pdf_y = 1.0/np.sqrt(2*np.pi*var_U)*np.exp(-0.5*(pdf_x-avg_U)**2/var_U)
#
#figure
# plt.figure(1)
# plt.hist(U_data,30,normed=True)
# plt.plot(pdf_x, pdf_y, 'k--')
# plt.title('U norm. distribution (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_U, sem_U, sigma_U))
# plt.xlabel('U values')
# plt.ylabel('Probability')
# plt.grid(True)
#
# ''' first_peak DISTRIBUTION - GAMMA '''
# #mean, var and std
# avg_PSP = np.mean(first_peak)
# var_PSP = np.var(first_peak)
# sigma_PSP = np.sqrt(var_PSP) #STD
# sem_PSP = sigma_PSP/np.sqrt(len(first_peak))
#
# k_PSP = (avg_PSP/sigma_PSP)**2
# Th_PSP = (sigma_PSP)**2/avg_PSP
#
# # From these two values we know the shape of the fitted Gamma
# pdf_x = np.linspace(np.min(first_peak), np.max(first_peak), 80)
# pdf_y = stats.gamma.pdf(pdf_x, a=k_PSP, scale=Th_PSP)
#
# #figure
# plt.figure(4)
# plt.hist(first_peak,20,normed=True)
# plt.plot(pdf_x, pdf_y, 'k--')
# plt.title('PSP distribution (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_PSP, sem_PSP, sigma_PSP))
# plt.xlabel('PSP values')
# plt.ylabel('Probability')
# plt.grid(True)
#
#
###########################################################################


# plt.figure(1)
# plt.xlabel('first EPSP amplitude')
# plt.ylabel('# connections')
# # plt.plot(EPSP2_x, EPSP2_y, 'g--')
# plt.hist([first_EPSP_amp,first_EPSP_amp_2], normed=True, label=['in Vitro', 'in Silico'])
# plt.legend()
#
# plt.figure(2)
# plt.xlabel('CV first EPSP amplitude')
# plt.ylabel('# connections')
# plt.hist([CV1_first,CV2_first], normed=True, label=['in Vitro', 'in Silico'])
# plt.legend()
# plt.show()

# plt.xlabel('stimulus index')
# plt.ylabel('CV')
# plt.plot(STIM_NUM, CVmean1, 'bo', label='in Vitro')
# #plt.plot(STIM_NUM, CVmean1, 'b--')
# plt.errorbar(STIM_NUM, CVmean1, yerr=CVesterr1, color='blue', fmt='--')
# plt.plot(STIM_NUM, CVmean2, 'go', label='in Silico')
# #plt.plot(STIM_NUM, CVmean2, 'g--')
# plt.errorbar(STIM_NUM, CVmean2, yerr=CVesterr2, color='green', fmt='--')
# plt.legend()
# plt.show()

''' BINOMIAL MODEL FITTING '''
# nb = 6.0
# qs = 0.2
# dv = []
# Pr = []
# # cv_bin = []
# # for v in np.arange(0.2,5.0,0.01):
# #     pr = v/nb*qs
# #     #print pr
# #     cv = np.sqrt((1-pr)/nb*pr)
# #     print 1-pr
# #     print nb*pr
# #     #print cv
# #     dv.append(v)
# #     cv_bin.append(cv)
#
# print 'first_EPSP_amp', first_EPSP_amp
# print 'CV', CV1_first
#
# for cv in CV1_first:
#     pr = 1/(cv**2*nb+1)
#     Pr.append(pr)
#     v = pr*nb*qs
#     dv.append(v)
#
# print np.mean(Pr)

# plt.xlabel('first EPSP amplitude')
# plt.ylabel('CV first EPSP amplitude')
# plt.plot(first_EPSP_amp_2, CV2_first, 'go', label='in Silico')
# plt.plot(first_EPSP_amp, CV1_first, 'bo', label='in Vitro')
# #plt.plot(dv, CV1_first, 'k--')
# plt.legend()
# plt.show()



''' pick a sample connection by name (ie: 'c4') '''
# # we make a loop over all the connections in raw_data
# #for l in range(1,101):
# file = 'noise_simulation_old_6.h5'
# raw_data = h5py.File(RAW_DATA_PATH)
# raw_data2 = h5py.File(RAW_DATA_PATH2+file)
#
# MAX = []
# MIN = []
#
# sample_connection1 = (raw_data['c60'].value)*1000.0
# sample_connection1 = np.transpose(sample_connection1)
# for sweep in sample_connection1:
#     max = np.max(sweep)
#     MAX.append(max)
#     min = np.min(sweep)
#     MIN.append(min)
#
# sample_connection2 = raw_data2['nrrp1'].value
# for sweep in sample_connection2:
#     max = np.max(sweep)
#     MAX.append(max)
#     min = np.min(sweep)
#     MIN.append(min)
#
# max_val = np.max(MAX)
# min_val = np.min(MIN)
#
# mean_samp_conn1 = np.mean(sample_connection1, axis=0)
# print mean_samp_conn1
# mean_samp_conn2 = np.mean(sample_connection2, axis=0)
#
# mean_samp_norm1 = (mean_samp_conn1 - min_val) / (max_val - min_val)
# mean_samp_norm2 = (mean_samp_conn2 - min_val) / (max_val - min_val)
#
#
# baseline = 0
# mean_samp_base1 = mean_samp_norm1 - mean_samp_norm1[0]
# mean_samp_base2 = mean_samp_norm2 - mean_samp_norm2[0]
#
# plt.figure(1, figsize=(7,5))
# plt.title('in Vitro trace 60 norm')
# plt.xlabel('time (s)')
# plt.ylabel('Voltage norm')
# plt.ylim([-0.2, 1])
#
# for sweep in sample_connection1:
#     # normalize sweep trace
#     sweep_norm = (sweep - min_val) / (max_val - min_val)
#     sweep_base = sweep_norm - sweep_norm[0]
#     plt.plot(time_vitro, sweep_base, color="grey", linewidth=0.1, linestyle="-")
#
# plt.plot(time_vitro, mean_samp_base1, color="blue", linewidth=3.0, linestyle="-")
# plt.grid(color='black', linestyle='-.', linewidth=0.5)
# plt.show()
#
# plt.figure(2, figsize=(7,5))
# plt.title('in Silico trace 6 norm')
# plt.xlabel('time (s)')
# plt.ylabel('Voltage norm')
# plt.ylim([-0.2, 1])
#
# for sweep in sample_connection2:
#     # normalize sweep trace
#     sweep_norm = (sweep - min_val) / (max_val - min_val)
#     sweep_base = sweep_norm - sweep_norm[0]
#     plt.plot(time_silico, sweep_base, color="grey", linewidth=0.1, linestyle="-")
#
# plt.plot(time_silico, mean_samp_base2, color="green", linewidth=3.0, linestyle="-")
# plt.grid(color='black', linestyle='-.', linewidth=0.5)
# plt.show()
# #plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/RESULTS/figures/InVitroTraces/inVitroTrace_%s.pdf' %c)
#
#     #plt.figure(2, figsize=(7,5))
#     #plt.plot(time_silico, mean_samp_connection)
#     #plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/inSilicoTraceOLD_10.pdf')


''' COMPUTE LATENCY, TAU_RISE AND AMPLITUDE OF FIRST EPSP '''
# first_EPSP_amp = []
# TAU_rise = []
# latency = []
# # we make a loop over all the connections in raw_data
# list_key = raw_data.keys()
# i = 0
# for con_num in list_key:
#     i = i + 1
#     if (con_num == 'c7' or con_num == 'c9' or con_num == 'c11'):  # in the case of this raw_data we don't want these three connections
#         pass
#     else:
#         print con_num
#
#         sample_connection = np.transpose(raw_data[con_num].value)
#         sample_connection = sample_connection[0:20]
#
#         EPSP_amp, tau_rise, lat = at.amp_rise_lat_firstEPSP(sample_connection, STIM_TIMES, time, t_wind_aft)
#
#         first_EPSP_amp.append(EPSP_amp)
#         TAU_rise.append(tau_rise)
#         latency.append(lat)
#
#
# print 'amplitude (mV)', first_EPSP_amp
# print 'MEAN amplitude (mV)', np.mean(first_EPSP_amp)
# print 'STD amplitude (mV)', np.std(first_EPSP_amp)
# print 'MAX amplitude (mV)', np.max(first_EPSP_amp)
# print 'MIN amplitude (mV)', np.min(first_EPSP_amp)
#
# print 'tau_rise (ms)', TAU_rise
# print 'MEAN tau_rise (ms)', np.mean(TAU_rise)
# print 'STD tau_rise (ms)', np.std(TAU_rise)
# print 'MAX tau_rise (ms)', np.max(TAU_rise)
# print 'MIN tau_rise (ms)', np.min(TAU_rise)
#
# print 'latency', latency
# print 'MEAN latency (ms)', np.mean(latency)
# print 'STD latency (ms)', np.std(latency)
# print 'MAX latency (ms)', np.max(latency)
# print 'MIN latency (ms)', np.min(latency)

''' RUN DECONVOLVE TRACE AND COMPUTE U, D AND F '''
# # we make a loop over all the connections in raw_data
# for con_num in list_key:
#     sample_connection = np.transpose(raw_data[con_num].value)
#
#     # compute mean trace
#     mean_t = np.mean(sample_connection, axis=0)
#
#     # filt mean trace
#     filt_mean_t = at.butter_lowpass_filter(mean_t - mean_t[0], 50, fs, order=5)
#
#     # deconvolve filtered mean trace
#     TAU_MEM = TAU_MEM_EACH[con_num]*10.0
#     dec_filt_mean_t = at.deconvolver(filt_mean_t, TAU_MEM)
#
#     # normalize deconvolved trace
#     max_dec_filt_mean_t = np.max(dec_filt_mean_t[1000:1300])
#     min_dec_filt_mean_t = np.min(dec_filt_mean_t[1000:1300])
#     dec_filt_mean_t_norm = (dec_filt_mean_t - min_dec_filt_mean_t) / (max_dec_filt_mean_t - min_dec_filt_mean_t)
#
#     # compute peaks for normalized trace
#     dec_peaks, dec_peaks_time = at.compute_peaks(dec_filt_mean_t_norm, STIM_TIMES, t_wind_aft)
#
#     # U, D and F estimation from peaks
#     GeneticAlgorithm.EXPERIMENTAL = dec_peaks
#     club = GeneticAlgorithm.main_ga()
#
#     U = club[0]
#     F = club[1]
#     D = club[2]
#     fitting_value = club[3]


'''''RUN JKK_traces and CV_JKK_trace for all connections'''''
# save cv list
# cv_jkk_array = []
# amp_jkk_array = []

# for con_num in list_key:
#     '''' pick a sample connection by name (ie: 'c60') and take no more than 20 traces '''''
#     sample_connection = raw_data[con_num].value
#     sample_connection = np.transpose(sample_connection)
#
#     mean_array, max_array, min_array, peak_time, amp_array, cv = at.JKK_traces(sample_connection, STIM_TIMES, t_wind_aft, fs, SIMULATION_TIME)
#
#     cv_jkk_array.append(cv)
#     amp_jkk_array.append(amp_array)
#
# # print len(cv_jkk_array)
# cv_jkk_mean = np.mean(cv_jkk_array, axis=0)
# CV_stderr = np.std(cv_jkk_array, axis=0) / np.sqrt(len(cv_jkk_array))
# MEAN = np.mean(cv_jkk_mean)
# STD = np.std(cv_jkk_mean)
#
# plt.figure()
# plt.title('Mean CV profile in vitro traces -JACKKNIFE: mean=%.2f std=%.2f' %(MEAN, STD))
# plt.xlabel('# stimulus')
# plt.ylabel('CV value')
# plt.plot(STIM_NUM, cv_jkk_mean, color="blue", linewidth=5.0, linestyle="-")
# plt.show()

