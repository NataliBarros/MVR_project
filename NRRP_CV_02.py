import h5py
import matplotlib.pyplot as plt
import numpy as np
import AnalyzeTrace as at
plt.style.use('seaborn-deep')

# path to the raw data
RAW_DATA_PATH = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Data_h5_files/invitro_raw.h5'

# total time of the recording
SIMULATION_TIME = 1.3
fs = 10000.0
#fs2 = 1.0/0.000025

# times where a stimulus is performed
STIM_TIMES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 10000]
#STIM_TIMES_silico = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 40000]
#stim_times_s = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 1.0]

# lists used in plots
STIM_NUM = [1,2,3,4,5,6,7,8,9]
#time_silico = np.arange(0,1.3,0.000025)
time_vitro = np.arange(0,1.3,0.0001)
t_wind_aft = 300
t_wind_bef = 50

# Rinput = 181.0
# # dictionary with tau_mem values for this specific data
# all
TAU_MEM_EACH = {'c0':20.9, 'c1':19.1, 'c2':30.4, 'c4':32.9, 'c10':33.3, 'c12':35.5, 'c13':43.5, 'c14':41.2, 'c31':31.7, 'c59':28.6, 'c60':46.9, 'c61':31.6, 'c62':23.0, 'c63':22.7, 'c64':34.1, 'c65':18.4, 'c66':29.1, 'c67':22.0, 'c68':37.1, 'c69':53.9, 'c70':18.7, 'c71':37.5, 'c73':36.7, 'c74':25.0, 'c75':39.0, 'c76':26.2, 'c77':48.0, 'c78':43.8, 'c79':17.1, 'c80':25.1, 'c81':25.6, 'c82':62.3, 'c83':46.7}
# Good
# TAU_MEM_EACH = {'c1':19.1, 'c2':30.4, 'c4':32.9, 'c12':35.5, 'c31':31.7, 'c60':46.9, 'c64':34.1, 'c66':29.1, 'c71':37.5, 'c73':36.7, 'c74':25.0, 'c75':39.0, 'c81':25.6}
# One
# TAU_MEM_EACH = {'c0':20.9}


''' ############### CV PROFILES IN VITRO ########################################## '''
# load raw invitro data
raw_data = h5py.File(RAW_DATA_PATH)

CV_connection1 = []
CV_connection2 = []
CV_connection3 = []

# # we make a loop over all the connections in raw_data
list_key = TAU_MEM_EACH.keys()

##### Compute CV -- no JKK no Deconv --
for i in range(len(list_key)):
    sample_connection = np.transpose(raw_data[list_key[i]].value)
    sample_connection = sample_connection

    amps = []
    for sc1 in sample_connection:
        #plt.figure(1)
        #plt.plot(sc)

        # normalize sc
        # scmax1 = np.max(sc1)
        # scmin1 = np.min(sc1)
        # sc_norm1 = (sc1 - scmin1)/(scmax1 - scmin1)
        # sc_norm1 = sc1 - sc1[0]

        max1, min1, amplitudes1 = at.compute_amplitude2(sc1, STIM_TIMES, t_wind_aft)
        amps.append(amplitudes1)

        # plt.figure(1)
        # plt.plot(sc_norm1)
        # plt.plot(STIM_TIMES, max1, 'ro')
        # plt.plot(STIM_TIMES, min1, 'go')
        # plt.show()

    amps_MEAN1 = np.mean(amps, axis = 0)
    print 'amp1', amps_MEAN1

    DIF1 = []
    for am in amps:
        dif1 = (am - amps_MEAN1)**2
        DIF1.append(dif1)

    print 'sum1', np.sum(DIF1, axis=0)
    N1 = np.float(len(sample_connection))
    amps_STD1 = np.sqrt(np.sum(DIF1, axis = 0)/N1)
    print 'std1', amps_STD1

    CV1 = amps_STD1/amps_MEAN1
    CV_connection1.append(CV1)

CV_vitro = np.mean(CV_connection1, axis = 0)
print 'CV raw', CV_vitro

##### Compute CV -- JKK --
for i in range(len(list_key)):
    sample_connection = np.transpose(raw_data[list_key[i]].value)
    sample_connection = sample_connection
    #print 'len sample_conn', len(sample_connection)

    amp_means = []
    for sweep_num in range(len(sample_connection)):
        # remove one different sweep from the trace set in each iteration and compute the mean
        new_sample = np.delete(sample_connection, sweep_num, 0)

        new_sample_mean = np.mean(new_sample, axis = 0)

        # normalize new_sample_mean
        # scmax1 = np.max(new_sample_mean)
        # scmin1 = np.min(new_sample_mean)
        # new_sample_mean_norm = (new_sample_mean - scmin1) / (scmax1 - scmin1)
        # new_sample_mean_norm = new_sample_mean - new_sample_mean[0]

        max2, min2, amplitudes2 = at.compute_amplitude2(new_sample_mean, STIM_TIMES, t_wind_aft)

        # plt.figure(2)
        # plt.plot(new_sample_mean_norm)
        # plt.plot(STIM_TIMES, max2, 'ro')
        # plt.plot(STIM_TIMES, min2, 'go')
        # plt.show()

        amp_means.append(amplitudes2)

    amp_MEAN2 = np.mean(amp_means, axis = 0)

    print 'amp2', amp_MEAN2

    DIF2 = []
    for am in amp_means:
        #print 'amp', am
        dif2 = (len(sample_connection-1)*(am - amp_MEAN2))**2
        #print 'dif2', dif2
        DIF2.append(dif2)

    N2 = np.float(len(sample_connection))
    print 'sum2', np.sum(DIF2, axis = 0)
    amp_std2 = np.sqrt(np.sum(DIF2, axis = 0)/N2)
    print 'std2', amp_std2

    CV2 = amp_std2/amp_MEAN2
    CV_connection2.append(CV2)

CV_vitroJKK = np.mean(CV_connection2, axis = 0)
print 'CV_vitro_JKK', CV_vitroJKK

##### Compute CV -- Deconvolve --
for i in range(len(list_key)):
    TAU = TAU_MEM_EACH[list_key[i]]*10.0
    sample_connection = np.transpose(raw_data[list_key[i]].value)
    sample_connection = sample_connection
    #print 'len sample_conn', len(sample_connection)

    amp_dec = []
    for sc2 in sample_connection:
        # filt sc
        fsc = at.butter_lowpass_filter(sc2 - sc2[0], 50, fs, order=5)
        dec_sc = at.deconvolver(fsc, TAU)

        #plt.figure(1)
        #plt.plot(dec_sc)

        # normalized dec_sc
        # scmax2 = np.max(dec_sc)
        # scmin2 = np.min(dec_sc)
        # dec_sc_norm = (dec_sc - scmin2) / (scmax2 - scmin2)
        # dec_sc_norm = dec_sc - dec_sc[0]

        max3, min3, amplitudes3 = at.compute_amplitude(dec_sc, STIM_TIMES, t_wind_bef, t_wind_aft)

        # plt.figure(3)
        # plt.plot(dec_sc_norm)
        # plt.plot(STIM_TIMES, max3, 'ro')
        # plt.plot(STIM_TIMES, min3, 'go')
        # plt.show()

        amp_dec.append(amplitudes3)

    amp_MEAN3 = np.mean(amp_dec, axis = 0)
    print 'amp3', amp_MEAN3

    DIF3 = []
    for am in amp_dec:
        dif3 = (am - amp_MEAN3)**2
        DIF3.append(dif3)

    N3 = np.float(len(sample_connection))
    print 'sum3', np.sum(DIF3, axis = 0)
    amp_std3 = np.sqrt(np.sum(DIF3, axis = 0)/N3)
    print 'std3', amp_std3

    CV3 = amp_std3/amp_MEAN3
    CV_connection3.append(CV3)

CV_vitroDec = np.mean(CV_connection3, axis = 0)
print 'CV_vitro_Dec', CV_vitroDec

plt.figure()
#plt.title('connection 80 - bad-')
plt.xlabel('stimulus')
plt.ylabel('CV')
plt.plot(CV_vitro, label = 'noJKKnoDec')
plt.plot(CV_vitroJKK, label = 'JKK')
plt.plot(CV_vitroDec, label = 'Dec')
plt.legend()
plt.show()