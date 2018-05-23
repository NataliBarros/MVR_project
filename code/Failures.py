# Natali barros Zulaica
# Dec - 2017

import h5py
import numpy as np
import matplotlib.pyplot as plt


RAW_DATA_PATH = 'h5_data/invitro_raw.h5'#'/gpfs/bbp.cscs.ch/project/proj35/MVR/invitro/preprocessing/invitro_raw.h5'
SIMULATION_TIME = 1.3 
STIM_TIMES = [1000,1500,2000,2500,3000,3500,4000,4500,10000]
stim_num = [1,2,3,4,5,6,7,8,9]
time = range(13001)

MEAN_PEAKS = []
FAIL_PERCENT = []

''' load the raw invitro data '''
raw_data = h5py.File(RAW_DATA_PATH)

list_key = raw_data.keys()
i = 0
for con_num in list_key:
    i = i + 1
    print con_num
    if (con_num == 'c7' or con_num =='c9' or con_num =='c11'):
        pass
    else:
        ''' pick a sample connection by name (ie: 'c60') and take up to 20 traces '''

        sample_connection = raw_data[con_num].value
        sample_connection = sample_connection.transpose()
        #sample_connection = sample_connection[0:20]
        trace_num = len(sample_connection)
        ''' make the mean of the traces eliminating each time one, and extract the peaks amplitude for each mean trace '''
        '''JACKKNIFE BOOTSTRAPING'''
        peaks_array=[] #here we will save the peaks for each mean trace
        for trace in range(trace_num):
            sample_con = sample_connection
            new_sample = np.delete(sample_con, trace, 0)
            mean_sample = np.mean(new_sample, axis=0)
            peaks = []
            for t in STIM_TIMES:
                pk = np.max(mean_sample[t:t+300])-np.min(mean_sample[t:t+300])
                peaks.append(pk)
            peaks_array.append(peaks)
            
        ''' compute mean amplitude for each peak '''
        mean_peaks_array = np.mean(peaks_array, axis=0)*1000.0
        MEAN_PEAKS = MEAN_PEAKS + mean_peaks_array.tolist()
        #MEAN_PEAKS.append(mean_peaks_array[0])
        #MEAN_PEAKS.append(mean_peaks_array[8])

        '''compute failures per trace in each connection'''
        n1 = 0 #counters for failures for each "epsp" after stimulation
        n2 = 0
        n3 = 0
        n4 = 0
        n5 = 0
        n6 = 0
        n7 = 0
        n8 = 0
        n9 = 0
        for i in range(trace_num):
            sample_connection_uni = []

            '''normalizing trace'''
            pk_max = np.max(sample_connection[i])
            pk_min = np.min(sample_connection[i])
            for z in sample_connection[i]:
                z_uni = (z-pk_min)/(pk_max-pk_min)
                sample_connection_uni.append(z_uni)
            mean_baseline = np.mean(sample_connection_uni, axis=0)
            #print sample_connection_uni

            '''compute noise as std of amplitudes of small peaks before the first EPSP'''
            noise_amp =[]
            noise_time = np.arange(50,1000,10)
            for t in noise_time:
                na = np.max(sample_connection[i][t:t+10]) - np.min(sample_connection[i][t:t+10])
                noise_amp.append(na)
            noise = np.std(noise_amp)
            failureline = 1.6*noise + mean_baseline

            #to plot the line -  create a vector with failure line value
            failure_vec = np.ones(13001)*failureline
            # plt.figure()
            # plt.title('inVitro conn_%s; trace_%s + fail threshold' %(con_num, i))
            # plt.plot(time, failure_vec, 'r--')
            # plt.plot(sample_connection_uni)
            # plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/TESTING-PROCEDURE/conn%s_trace%s.png' %(con_num, i))

            '''compute peaks for each trace. If peak bellow failureline then is a failure'''
            peak_trace=[]
            count = 0
            for t in STIM_TIMES:
                count = count+1
                pk = np.max(sample_connection_uni[t:t+300])
                #print pk
                if pk <= failureline:
                    if count==1:
                        n1 = n1 + 1
                    if count==2:
                        n2 = n2 + 1
                    if count==3:
                        n3 = n3 + 1
                    if count==4:
                        n4 = n4 + 1
                    if count==5:
                        n5 = n5 + 1
                    if count==6:
                        n6 = n6 + 1
                    if count==7:
                        n7 = n7 + 1
                    if count==8:
                        n8 = n8 + 1
                    if count==9:
                        n9 = n9 + 1
                else:
                    pass
        fail_percent = [n1*100.0/float(trace_num), n2*100.0/float(trace_num), n3*100.0/float(trace_num), n4*100.0/float(trace_num), n5*100.0/float(trace_num), n6*100.0/float(trace_num), n7*100.0/float(trace_num), n8*100.0/float(trace_num), n9*100.0/float(trace_num)]

        # plt.figure()
        # plt.title('percentage of failure each peak -con_%s' %con_num)
        # plt.xlabel('# stimulus')
        # plt.ylabel('Percentage')
        # plt.bar(stim_num, fail_percent, align='center')
        # plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/TESTING-PROCEDURE/failures_%s' %con_num)
        FAIL_PERCENT.append(fail_percent)
        #FAIL_PERCENT = FAIL_PERCENT + fail_percent

mean_fails = np.mean(FAIL_PERCENT, axis=0)
print np.mean(FAIL_PERCENT, axis=0)


plt.figure()
plt.title('Percentage of failures - threshold = noise*1.6 \n percentages = %s' %mean_fails)
plt.bar(stim_num, mean_fails, align='center')
plt.xlabel('Stimulus')
plt.ylabel('Mean %Failures')
plt.show()
# plt.savefig('/home/barros/Desktop/Project_MVR/MVR_warmupProject/TESTING-PROCEDURE/failures_th1.6.png')
