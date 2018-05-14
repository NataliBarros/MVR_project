# Natali barros Zulaica
# Jan - 2018

import h5py
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def CV_Markram_TracebyTrace(sample_connection, STIM_TIMES, t_wind_aft, a):
    """
    This function computes the coeficient of variation (cv) of each EPSP in each trace in sample_connection
    and the noise baseline based on Markram, 1997
    :param sample_connection: array with voltage traces
    :param STIM_TIMES:  list of int with time vector indexes where a stimulus happens
    :param t_wind_aft: int value for a time window after the stimulus to find the max value
    :return amp_mean: list with each peak amplitude computed as the mean of each amplitude subtracted from each trace in a sample
    :return amp_std: list with each peak std computed as the std of the amplitudes subtracted from each trace in a sample
    :return amp_CV: list with the cv values for each peak
    """
    EPSP_array = [] # safe all amplitudes for each trace as a list of lists
    baseline_array = []
    amplitude = []
    count = 0
    for trace in sample_connection:
        count = count + 1
        # compute NOISE as std of amplitudes of small peaks before the first EPSP; compute baseline noise
        # define range before first EPSP as baseline
        # IN VITRO
        # baseline_noise = trace[50:STIM_TIMES[0] - 50]
        # baseline_voltage = trace[500:STIM_TIMES[0]]
        # IN SILICO
        baseline_noise = trace[200:STIM_TIMES[0] - 200]
        ######
        baseline_voltage = trace[2000:STIM_TIMES[0]]
        mean_baseline = np.mean(baseline_voltage)
        std_baseline = np.std(baseline_voltage)
        ######
        mean_baseline_large = []
        # IN VITRO
        # for i in np.arange(50, len(baseline_noise) + 50):
        #     mean_baseline_large.append(mean_baseline)
        # IN SILICO
        for i in np.arange(200, len(baseline_noise) + 200):
            mean_baseline_large.append(mean_baseline)
        #baseline_array.append(mean_baseline)
        noise_max = []
        noise_min = []
        noise_amp = []
        # IN VITRO
        #noise_time = np.arange(50, len(baseline_noise)+50, 10)
        # IN SILICO
        noise_time = np.arange(200, len(baseline_noise) + 200, 40)
        for t in noise_time:
            # IN VITRO
            # na = np.max(trace[t:t+10])-np.min(trace[t:t+10])
            # noise_max.append(np.max(trace[t:t+10]))
            # noise_min.append(np.min(trace[t:t+10]))
            # IN SILICO
            na = np.max(trace[t:t + 40]) - np.min(trace[t:t + 40])
            noise_max.append(np.max(trace[t:t + 40]))
            noise_min.append(np.min(trace[t:t + 40]))
            noise_amp.append(na)

        # check max and min for the baseline ...
        # plt.figure()
        # plt.plot(np.arange(0,len(trace)),trace)
        # plt.plot(np.arange(200,len(baseline_noise)+200), baseline_noise)
        # plt.plot(noise_time, noise_max, 'r.')
        # plt.plot(noise_time, noise_min, 'g.')
        # plt.show()

        # define noise
        NOISE = np.std(noise_amp)
        #baseline_noise = np.mean(noise_amp)
        #baseline_noise_array.append(baseline_noise)

        # compute max peak value for the first EPSP as an overage of -5 and +5 points around the max
        max_value = np.max(trace[STIM_TIMES[0]:STIM_TIMES[0] + t_wind_aft])
        min_value = np.min(trace[STIM_TIMES[0]:STIM_TIMES[0] + t_wind_aft])
        #time = range(13000)
        time = range(52000)
        #print 'MAX', max_value
        for v, i in zip(trace[STIM_TIMES[0]:STIM_TIMES[0] + t_wind_aft], time[STIM_TIMES[0]:STIM_TIMES[0] + t_wind_aft]):
            if v == max_value:
                #EPSP_time = np.arange(i-5, i+5)
                EPSP_time = np.arange(i-5, i+5)
                time2 = i
                #EPSP = trace[i-5:i+5]
                #EPSP_mean = np.mean(trace[i-5:i+5])
                EPSP = trace[i-20:i+20]
                EPSP_mean = np.mean(trace[i-20:i+20])

        # plt.figure()
        # plt.xlabel('time')
        # plt.ylabel('voltage')
        # plt.plot(time[0:6000], trace[0:6000])
        # plt.plot(EPSP_time,EPSP, 'r.')
        # plt.plot(time2, EPSP_mean, 'cs')
        # plt.plot(np.arange(200, len(baseline_noise) + 200), mean_baseline_large, 'g--')
        # plt.plot(np.arange(200, len(baseline_noise) + 200), baseline_noise)
        # plt.plot(noise_time, noise_max, 'm.')
        # plt.plot(noise_time, noise_min, 'y.')
        # #plt.show()
        # plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/TESTING-PROCEDURE/InSilico_Amplitude_Markram/amp_Markram_SIM%strace%s.png' %(a, count))

        amp = np.abs(EPSP_mean - np.mean(baseline_noise))#*1000.0 #---ONLY FOR IN VITRO
        EPSP_array.append(EPSP)
        baseline_array.append(mean_baseline)
        amplitude.append(amp)

    AMP = np.mean(amplitude)
    std_AMP = np.std(amplitude)

    '''compute CV corrected by subtraction of baseline variation to EPSP variation'''
    EPSP_var = np.var(amplitude)
    baseline_var = np.var(baseline_array)
    cv_corrected = np.abs(EPSP_var-baseline_var)
    #CV = std_AMP/AMP
    CV = np.sqrt(cv_corrected)/AMP

    return NOISE, AMP, std_AMP, CV, std_baseline


############### RUN --------------------------------------------------------------------------------------

RAW_DATA_PATH = '/home/barros/Desktop/Project_MVR/MVR_warmupProject/h5_data/invitro_raw.h5'
SIM_DATA_PATH = '/home/barros/Desktop/Project_MVR/MVR_warmupProject/h5_data/noise_simulation_new/'
RAW_DATA_PATH2 = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/noise_simulation_new01/'
#'/gpfs/bbp.cscs.ch/project/proj35/MVR/invitro/preprocessing/invitro_raw.h5'

SIMULATION_TIME = 1.3
STIM_TIMES = [1000,1500,2000,2500,3000,3500,4000,4500,10000]
STIM_TIMES_silico = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 40000]
#stim_num = [1,2,3,4,5,6,7,8,9]

#t_wind_aft = 300
t_wind_aft = 1200

''' load the raw invitro data '''
#raw_data = h5py.File(SIM_DATA_PATH)

EPSP_AMP = []
EPSP_CV = []
EPSP_STD = []

STD_all = []
NOISE_all = []

# # IN VITRO -------------------------------------------------------------------------------------
# list_key = raw_data.keys()
# #list_key = ['c77']
# i = 0
# for con_num in list_key:
#     i = i + 1
#     #print con_num
#     if (con_num == 'c7' or con_num =='c9' or con_num =='c11'):
#         pass
#     else:
#         ''' pick a sample connection by name (ie: 'c60') '''
#         sample_connection = raw_data[con_num].value
#         sample_connection = sample_connection.transpose()
#
#         NOISE, AMP, std_AMP, CV, STD = CV_Markram_TracebyTrace(sample_connection, STIM_TIMES, t_wind_aft)
#         EPSP_AMP.append(AMP)
#         EPSP_STD.append(std_AMP)
#         EPSP_CV.append(CV)
#         STD_all.append(STD)
#         NOISE_all.append(NOISE)

### IN SILICO --------------------------------------------------------------------------------------
# for a in range(1, 101):
#     file = 'noise_simulation_%s.h5' %a
#     path_complete = SIM_DATA_PATH + file
#     print path_complete
#     raw_data2 = h5py.File(path_complete)
#     sample_connection = raw_data2['nrrp3'].value
#
#     NOISE, AMP, std_AMP, CV, STD = CV_Markram_TracebyTrace(sample_connection, STIM_TIMES_silico, t_wind_aft, a)
#
#     EPSP_AMP.append(AMP)
#     EPSP_STD.append(std_AMP)
#     EPSP_CV.append(CV)
#     STD_all.append(STD)
#     NOISE_all.append(NOISE)
#
#
# print np.mean(NOISE_all)
# print np.mean(STD_all)
#
# print EPSP_AMP
# print 'mean', np.mean(EPSP_AMP)
# print 'max EPSP', np.max(EPSP_AMP)
# print 'min EPSP', np.min(EPSP_AMP)
# print 'std', np.mean(EPSP_STD)
# print 'cv', np.mean(EPSP_CV)
# print 'cv min', np.min(EPSP_CV)
# print 'cv max', np.max(EPSP_CV)
# print 'std_cv', np.std(EPSP_CV)

EPSP_arr_dic = {}
CV_arr_dic = {}


lambda_values = np.arange(0.0, 12.9, 0.1)

for l in lambda_values:
    print 'computing CV for lambda = %.1f' %l
    #NRRPdic['%.1f' %l]=[]
    EPSP_arr_dic['%.1f' % l]=[]
    CV_arr_dic['%.1f' % l]=[]
    nrrp = poisson.rvs(l, size=100, loc=1)
    #print nrrp
    nrrp_mean = np.mean(nrrp)
    nrrp_std = np.std(nrrp)
    #NRRPdic['%.1f' %l].append(nrrp_mean)
    #NRRPdic['%.1f' %l].append(nrrp_std)

    for a, n in zip(range(1, 101), nrrp):
        file = 'noise_simulation_%s.h5' % a
        raw_data2 = h5py.File(RAW_DATA_PATH2 + file)
        if n > 24:
            sample_connection2 = raw_data2['nrrp24'].value
        else:
            sample_connection2 = raw_data2['nrrp%s' % n].value

        NOISE, AMP, std_AMP, CV, STD = CV_Markram_TracebyTrace(sample_connection2, STIM_TIMES_silico, t_wind_aft, a)

        EPSP_arr_dic['%.1f' % l].append(AMP)
        CV_arr_dic['%.1f' % l].append(CV)

EPSP = []
CV = []
epsp_vitro = []
epsp_vitro_up = []
epsp_vitro_down = []
cv_mk = []
cv_vitro = []

for l in lambda_values:
    cv_mk.append(0.52)
    cv_vitro.append(0.41)
    EPSP_mean = np.mean(EPSP_arr_dic['%.1f' % l])
    EPSP.append(EPSP_mean)
    CV_mean = np.mean(CV_arr_dic['%.1f' % l])
    CV.append(CV_mean)

plt.figure()
plt.title('CV vs lambda')
plt.xlabel('lambda')
plt.ylabel('CV')
plt.plot(lambda_values, CV, 'o')
plt.plot(lambda_values, cv_mk, 'r--')
plt.plot(lambda_values, cv_vitro, 'g--')
plt.show()



