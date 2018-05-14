#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:40:49 2017

@author: natalibarros
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson

''' CV VALUES FROM BIBLIOGRAPHY FOR EACH DIFFERENT CONNECTION '''
#''' ['CONNECTION_NAME', mean_cv, sd_cv] '''
#cv_literature = [['L4_SSC-L23_PC',0.27,0.13], ['L5_TPC:C-L5_TPC:C',0.58,0.24], ['L5_UPC-L5_UPC',0.58,0.24], ['L5_TTPC:A-L5_TTPC:B',0.52,0.37],
#              ['L4_SSC-L4_SSC',0.37,0.18], ['L4_SSC-L5_TPC:C',0.33,0.2], ['L5_TPC-L5_SBC',0.32,0.08]]

file_path = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/connsummaries/E2/connsummary_L23_PC-L23-PC__noise02.csv'

# num of repetitions for optimization
N = 50

# data from Srikant table
CV_ref = [0.33, 0.18]
PSP_ref = [0.9, 0.7]

# load data
data = pd.read_csv(file_path)

# define lambda values
lambda_values = np.arange(0.0, 12.9, 0.1)

# find num of connections in file
grouped = data.groupby(['pregid', 'postgid'])
nconn = len(grouped)

# Find the desired values
EPSP_mean_dic = {}
EPSP_mean_std_dic = {}
EPSP_CV_dic = {}
EPSP_CV_std_dic = {}
NRRP_dic = {}
NRRP_std_dic = {}

# Create the dictionaries to save values
for rep in range(N):
    EPSP_mean_dic['%s' %rep] = []
    EPSP_mean_std_dic['%s' %rep] = []
    EPSP_CV_dic['%s' %rep] = []
    EPSP_CV_std_dic['%s' %rep] = []
    NRRP_dic['%s' %rep] = []
    NRRP_std_dic['%s' %rep] = []

# chose cv values according to nrrp distribution
for rep in range(N):
    print 'computing repetition %s' %rep
    EPSP_mean = []
    EPSP_mean_std = []
    EPSP_CV = []
    EPSP_CV_std = []
    for l in lambda_values:
        # print 'computing CV for lambda = %.1f' % l
        # Define nrrp values as a poisson distribution
        nrrp = poisson.rvs(l, size=nconn, loc=1)
        nrrp_mean = np.mean(nrrp)
        nrrp_std = np.std(nrrp)
        NRRP_dic['%s' %rep].append(nrrp_mean)
        NRRP_std_dic['%s' %rep].append(nrrp_std)
        n = 0
        epsp_mean = []
        epsp_cv = []
        for stim, nrrp_val, d in zip(data['StimID'], data['Nrrp'], range(len(data['Nrrp']))):
            if stim == 0 and n < nconn and nrrp_val == nrrp[n]:
                epsp_mean.append(data['EPSP mean'][d])
                if np.isnan(data['EPSP CV'][d]) == True:
                    print 'Found CV = NaN'
                    data['EPSP CV'][d] = 0.0
                    epsp_cv.append(data['EPSP CV'][d])
                else:
                    epsp_cv.append(data['EPSP CV'][d])
                    n = n+1

        EPSP_mean.append(np.mean(epsp_mean))
        EPSP_mean_std.append(np.std(epsp_mean))
        EPSP_CV.append(np.mean(epsp_cv))
        EPSP_CV_std.append(np.std(epsp_cv))

    EPSP_mean_dic['%s' % rep].append(EPSP_mean)
    EPSP_mean_std_dic['%s' % rep].append(EPSP_mean_std)
    EPSP_CV_dic['%s' % rep].append(EPSP_CV)
    EPSP_CV_std_dic['%s' % rep].append(EPSP_CV_std)

# Transpose dictionaries
for rep in range(N):
    EPSP_mean_dic['%s' % rep] = np.transpose(EPSP_mean_dic['%s' % rep])
    EPSP_mean_std_dic['%s' % rep] = np.transpose(EPSP_mean_std_dic['%s' % rep])
    EPSP_CV_dic['%s' % rep] = np.transpose(EPSP_CV_dic['%s' % rep])
    EPSP_CV_std_dic['%s' % rep] = np.transpose(EPSP_CV_std_dic['%s' % rep])

# Find the closest value to the one in literature
CV_literature = np.ones(len(lambda_values))*CV_ref[0]
CV_distance = {}
CV_distance_min = []
for rep in range(N):
    CV_distance['%s' % rep] = []
    dif1 = np.abs(EPSP_CV_dic['%s' % rep]-CV_literature)
    #print 'dif', dif1
    CV_distance['%s' % rep].append(dif1)
    MIN_value = np.min(CV_distance['%s' % rep])
    if np.isnan(MIN_value) == True:
        pass
    else:
        CV_distance_min.append(MIN_value)

# that's the closest one
CV_distance_MIN = np.min(CV_distance_min)
print 'MIN', CV_distance_MIN

for rep in range(N):
    for l in range(len(lambda_values)):
        dif2 = np.abs(EPSP_CV_dic['%s' % rep][l] - CV_ref[0])
        if dif2 == CV_distance_MIN:
            print 'FOUND'
            PSP_sol = EPSP_mean_dic['%s' % rep][l]
            PSP_sol_std = EPSP_mean_std_dic['%s' % rep][l]
            CV_sol = EPSP_CV_dic['%s' % rep][l]
            CV_sol_std = EPSP_CV_std_dic['%s' % rep][l]
            NRRP_sol = NRRP_dic['%s' %rep][l]
            NRRP_sol_std = NRRP_std_dic['%s' %rep][l]
            print 'PSP', PSP_sol, 'PSP std', PSP_sol_std
            print 'CV', CV_sol, 'CV std', CV_sol_std
            print 'NRRP', NRRP_sol, 'NRRP std', NRRP_sol_std
        else:
            pass





# Plot results
#CV_literature = np.ones(len(lambda_values))*CV_ref[0]
CV_std_literature_up = np.ones(len(lambda_values))*(CV_ref[0]+CV_ref[1])
CV_std_literature_down = np.ones(len(lambda_values))*(CV_ref[0]-CV_ref[1])

#CV_vitro = np.ones(len(lambda_values))*(0.31) # only for connection L5_TPC-L5_TPC

PSP_literature = np.ones(len(lambda_values))*PSP_ref[0]


plt.figure(figsize=(9,7))
plt.title('CV vs lambda - L23_TPC-L23_TPC - \n CV=%.2f$\pm$%.4f \n NRRP=%.2f$\pm$%.4f' %(CV_sol,CV_sol_std,NRRP_sol,NRRP_sol_std))
plt.xlabel('lambda')
plt.ylabel('CV')
for rep in range(N):
    plt.plot(lambda_values, EPSP_CV_dic['%s' %rep], '.')
#plt.plot(lambda_values, CV_tot, 'b--')
plt.plot(lambda_values, CV_literature, 'r--', label='literature')
plt.plot(lambda_values, CV_std_literature_up, 'k--', alpha=0.5)
plt.plot(lambda_values, CV_std_literature_down, 'k--', alpha=0.5)
#plt.plot(lambda_values, CV_vitro, 'g--', label='vitro') # only for connection L5_TPC-L5_TPC
plt.legend()

# plt.figure(2)
# plt.title('EPSP vs lambda - L5_TPC-L5_SBC -')
# plt.xlabel('lambda')
# plt.ylabel('PSP')
# for rep in range(N):
#     plt.plot(lambda_values, EPSP_mean_dic['%s' %rep], '.')
# plt.plot(lambda_values, PSP_literature, 'r--', label='literature value')
# plt.legend()

# plt.figure(3)
# plt.title('NRRP vs lambda - L5_TPC-L5_SBC -')
# plt.xlabel('lambda')
# plt.ylabel('NRRP')
# for rep in range(50):
#     plt.plot(lambda_values, np.transpose(NRRP_dic['%s' %rep]), '.')
plt.show()




## MAKE DICTIONARIES ACCORDING TO LAMBDA
# for l in lambda_values:
#     NRRP_dic['%.1f' % l] = []
#     EPSP_mean_dic['%.1f' % l] = []
#     EPSP_CV_dic['%.1f' % l] = []
#
# for rep in range(2):
#     print 'computing repetition %s' %rep
#     for l in lambda_values:
#         #print 'computing CV for lambda = %.1f' % l
#
#         # Define nrrp values as a poisson distribution
#         nrrp = poisson.rvs(l, size=nconn, loc=1)
#         #print 'nrrp', nrrp
#         nrrp_mean = np.mean(nrrp)
#         #nrrp_std = np.std(nrrp)
#         NRRP_dic['%.1f' %l].append(nrrp_mean)
#         #NRRP_dic['%.1f' %l].append(nrrp_std)
#         EPSP_mean = []
#         EPSP_CV = []
#         n = 0
#         for stim, nrrp_val, d in zip(data['StimID'],data['Nrrp'],range(len(data['Nrrp']))):
#             #print 'nrrp[n]', nrrp[n]
#             if stim == 0 and n < nconn and nrrp_val == nrrp[n]:
#                 #print 'EPSP mean', data['EPSP mean'][d]
#                 #print 'EPSP cv', data['EPSP CV'][d]
#                 EPSP_mean.append(data['EPSP mean'][d])
#                 if np.isnan(data['EPSP CV'][d]) == True :
#                     data['EPSP CV'][d]=0
#                 else:
#                     EPSP_CV.append(data['EPSP CV'][d])
#                     n = n+1
#
#         #print 'EPSP_mean', EPSP_mean
#
#         EPSP_mean_dic['%.1f' % l].append(np.mean(EPSP_mean))
#         EPSP_CV_dic['%.1f' % l].append(np.mean(EPSP_CV))





