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

file_path = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Generalization/connsummaries/E2/connsummary_L5_TTPC_A-L5_TTPC_B_noise03_02.csv'
file_path_jkk = '/Users/natalibarros/Desktop/EPFL_BBP/MVR_warmupProject/Generalization/connsummaries/E2/connsummary_L5_TTPC_A-L5_TTPC_B_ns03_JKK_02.csv'
# num of repetitions for optimization
N = 50

# data from Srikant table
CV_ref = [0.31, 0.16]
PSP_ref = [1.47, 0.87]

# load data
data = pd.read_csv(file_path)
data_jkk = pd.read_csv(file_path_jkk)

# define lambda values
lambda_values = np.arange(0.1, 12.9, 0.1)

# find num of connections in file
grouped = data.groupby(['pregid', 'postgid'])
#print grouped
nconn = len(grouped)

# Find the desired values
EPSP_mean_dic = {}
EPSP_mean_std_dic = {}
EPSP_CV_dic = {}
EPSP_CV_std_dic = {}
NRRP_dic = {}
NRRP_std_dic = {}

# Find the desired values -JKK-
EPSP_mean_dic_jkk = {}
EPSP_mean_std_dic_jkk = {}
EPSP_CV_dic_jkk = {}
EPSP_CV_std_dic_jkk = {}
NRRP_dic_jkk = {}
NRRP_std_dic_jkk = {}

# Create the dictionaries to save values
for rep in range(N):
    EPSP_mean_dic['%s' %rep] = []
    EPSP_mean_std_dic['%s' %rep] = []
    EPSP_CV_dic['%s' %rep] = []
    EPSP_CV_std_dic['%s' %rep] = []
    NRRP_dic['%s' %rep] = []
    NRRP_std_dic['%s' %rep] = []

    EPSP_mean_dic_jkk['%s' %rep] = []
    EPSP_mean_std_dic_jkk['%s' %rep] = []
    EPSP_CV_dic_jkk['%s' %rep] = []
    EPSP_CV_std_dic_jkk['%s' %rep] = []
    NRRP_dic_jkk['%s' %rep] = []
    NRRP_std_dic_jkk['%s' %rep] = []

# chose cv values according to nrrp distribution of size 100
for rep in range(N):
    print 'computing repetition %s' %rep
    EPSP_mean = []
    EPSP_mean_std = []
    EPSP_CV = []
    EPSP_CV_std = []

    EPSP_mean_jkk = []
    EPSP_mean_std_jkk = []
    EPSP_CV_jkk = []
    EPSP_CV_std_jkk = []
    for l in lambda_values:
        # print 'computing CV for lambda = %.1f' % l
        # Define nrrp values as a poisson distribution
        nrrp = poisson.rvs(l, size=100, loc=1)
        nrrp_mean = np.mean(nrrp)
        nrrp_std = np.std(nrrp)
        NRRP_dic['%s' %rep].append(nrrp_mean)
        NRRP_std_dic['%s' %rep].append(nrrp_std)
        NRRP_dic_jkk['%s' % rep].append(nrrp_mean)
        NRRP_std_dic_jkk['%s' % rep].append(nrrp_std)
        n1 = 0
        n2 = 0
        epsp_mean = []
        epsp_cv = []
        epsp_mean_jkk = []
        epsp_cv_jkk = []

        # NO JKK
        for stim1, nrrp_val1, d1 in zip(data['StimID'], data['Nrrp'], range(len(data['Nrrp']))):
            if stim1 == 0 and n1 < 100 and nrrp_val1 == nrrp[n1]:
                epsp_mean.append(data['EPSP mean'][d1])
                if np.isnan(data['EPSP CV'][d1]) == True:
                    print 'Found CV = NaN'
                    data['EPSP CV'][d1] = 0.0
                    epsp_cv.append(data['EPSP CV'][d1])
                else:
                    epsp_cv.append(data['EPSP CV'][d1])
                    n1 = n1+1

        EPSP_mean.append(np.mean(epsp_mean))
        EPSP_mean_std.append(np.std(epsp_mean))
        EPSP_CV.append(np.mean(epsp_cv))
        EPSP_CV_std.append(np.std(epsp_cv))

        # JKK
        for stim2, nrrp_val2, d2 in zip(data_jkk['StimID'], data_jkk['Nrrp'], range(len(data_jkk['Nrrp']))):
            if stim2 == 0 and n2 < 100 and nrrp_val2 == nrrp[n2]:
                epsp_mean_jkk.append(data_jkk['EPSP mean'][d2])
                if np.isnan(data_jkk['EPSP CV'][d2]) == True:
                    print 'Found CV = NaN'
                    data_jkk['EPSP CV'][d2] = 0.0
                    epsp_cv_jkk.append(data_jkk['EPSP CV'][d2])
                else:
                    epsp_cv_jkk.append(data_jkk['EPSP CV'][d2])
                    n2 = n2+1

        EPSP_mean_jkk.append(np.mean(epsp_mean_jkk))
        EPSP_mean_std_jkk.append(np.std(epsp_mean_jkk))
        EPSP_CV_jkk.append(np.mean(epsp_cv_jkk))
        EPSP_CV_std_jkk.append(np.std(epsp_cv_jkk))

    EPSP_mean_dic['%s' % rep].append(EPSP_mean)
    EPSP_mean_std_dic['%s' % rep].append(EPSP_mean_std)
    EPSP_CV_dic['%s' % rep].append(EPSP_CV)
    EPSP_CV_std_dic['%s' % rep].append(EPSP_CV_std)

    EPSP_mean_dic_jkk['%s' % rep].append(EPSP_mean_jkk)
    EPSP_mean_std_dic_jkk['%s' % rep].append(EPSP_mean_std_jkk)
    EPSP_CV_dic_jkk['%s' % rep].append(EPSP_CV_jkk)
    EPSP_CV_std_dic_jkk['%s' % rep].append(EPSP_CV_std_jkk)

# Transpose dictionaries
for rep in range(N):
    EPSP_mean_dic['%s' % rep] = np.transpose(EPSP_mean_dic['%s' % rep])
    EPSP_mean_std_dic['%s' % rep] = np.transpose(EPSP_mean_std_dic['%s' % rep])
    EPSP_CV_dic['%s' % rep] = np.transpose(EPSP_CV_dic['%s' % rep])
    EPSP_CV_std_dic['%s' % rep] = np.transpose(EPSP_CV_std_dic['%s' % rep])
    # -JKK-
    EPSP_mean_dic_jkk['%s' % rep] = np.transpose(EPSP_mean_dic_jkk['%s' % rep])
    EPSP_mean_std_dic_jkk['%s' % rep] = np.transpose(EPSP_mean_std_dic_jkk['%s' % rep])
    EPSP_CV_dic_jkk['%s' % rep] = np.transpose(EPSP_CV_dic_jkk['%s' % rep])
    EPSP_CV_std_dic_jkk['%s' % rep] = np.transpose(EPSP_CV_std_dic_jkk['%s' % rep])

# # Find the closest value to the one in literature as the mean of all the min values
CV_literature = np.ones(len(lambda_values))*CV_ref[0]
CV_distance = {}
CV_distance_min = []
CV_distance_jkk = {}
CV_distance_min_jkk = []
for rep in range(N):
    CV_distance['%s' % rep] = []
    CV_distance_jkk['%s' % rep] = []
    dif1 = np.abs(EPSP_CV_dic['%s' % rep]-CV_literature)
    dif1_jkk = np.abs(EPSP_CV_dic_jkk['%s' % rep] - CV_literature)
    #print 'dif', dif1
    CV_distance['%s' % rep].append(dif1)
    CV_distance_jkk['%s' % rep].append(dif1_jkk)
    MIN_value = np.min(CV_distance['%s' % rep])
    MIN_value_jkk = np.min(CV_distance_jkk['%s' % rep])
    if np.isnan(MIN_value) == True:
        pass
    if np.isnan(MIN_value_jkk) == True:
        pass
    else:
        CV_distance_min.append(MIN_value)
        CV_distance_min_jkk.append(MIN_value_jkk)

NRRP_min = []
NRRP_std_min = []
CV_min = []
CV_std_min = []

NRRP_min_jkk = []
NRRP_std_min_jkk = []
CV_min_jkk = []
CV_std_min_jkk = []

for rep in range(N):
    for l in range(len(lambda_values)):
        dif2 = np.abs(EPSP_CV_dic['%s' % rep][l] - CV_ref[0])
        dif2_jkk = np.abs(EPSP_CV_dic_jkk['%s' % rep][l] - CV_ref[0])
        if dif2 in CV_distance_min:
            print 'FOUND'
            CV_min.append(EPSP_CV_dic['%s' % rep][l])
            CV_std_min.append(EPSP_CV_std_dic['%s' % rep][l])
            NRRP_min.append(NRRP_dic['%s' %rep][l])
            NRRP_std_min.append(NRRP_std_dic['%s' %rep][l])
        if dif2_jkk in CV_distance_min_jkk:
            print 'FOUND JKK'
            CV_min_jkk.append(EPSP_CV_dic_jkk['%s' % rep][l])
            CV_std_min_jkk.append(EPSP_CV_std_dic_jkk['%s' % rep][l])
            NRRP_min_jkk.append(NRRP_dic_jkk['%s' %rep][l])
            NRRP_std_min_jkk.append(NRRP_std_dic_jkk['%s' %rep][l])
        else:
            pass

CV_MIN = np.mean(CV_min)
CV_std_MIN = np.mean(CV_std_min)
NRRP_MIN = np.mean(NRRP_min)
NRRP_std_MIN = np.mean(NRRP_std_min)

CV_MIN_jkk = np.mean(CV_min_jkk)
CV_std_MIN_jkk = np.mean(CV_std_min_jkk)
NRRP_MIN_jkk = np.mean(NRRP_min_jkk)
NRRP_std_MIN_jkk = np.mean(NRRP_std_min_jkk)

print 'CV', CV_MIN
print 'std CV', CV_std_MIN
print 'nrrp', NRRP_MIN
print 'std nrrp', NRRP_std_MIN

print 'CV jkk', CV_MIN_jkk
print 'std CV jkk', CV_std_MIN_jkk
print 'nrrp jkk', NRRP_MIN_jkk
print 'std nrrp jkk', NRRP_std_MIN_jkk

### PLOTS

# Plot CV no JKK vs CV JKK
plt.figure(1)
plt.title('CV vs CV_JKK - L5_TPC:A-L5_TPC:B -')
plt.xlabel('CV')
plt.ylabel('CV_JKK')
for rep in range(N):
    plt.plot(EPSP_CV_dic['%s' %rep], EPSP_CV_dic_jkk['%s' %rep], '.')

# plot cv vs lambda for no-jkk and jkk
CV_literature = np.ones(len(lambda_values))*CV_ref[0]
CV_std_literature_up = np.ones(len(lambda_values))*(CV_ref[0]+CV_ref[1])
CV_std_literature_down = np.ones(len(lambda_values))*(CV_ref[0]-CV_ref[1])

plt.figure(2)
plt.title('CV vs lambda - L5_TPC:A-L5_TPC:B - \n CV=%.2f$\pm$%.4f \n NRRP=%.2f$\pm$%.4f' %(CV_MIN,CV_std_MIN,NRRP_MIN,NRRP_std_MIN))
plt.xlabel('lambda')
plt.ylabel('CV')
for rep in range(N):
    plt.plot(lambda_values, EPSP_CV_dic['%s' %rep], '.')
plt.plot(lambda_values, CV_literature, 'r--', label='literature')
plt.plot(lambda_values, CV_std_literature_up, 'k--', alpha=0.5)
plt.plot(lambda_values, CV_std_literature_down, 'k--', alpha=0.5)
plt.legend()

plt.figure(3)
plt.title('CV vs lambda - L5_TPC:A-L5_TPC:B JKK - \n CV=%.2f$\pm$%.4f \n NRRP=%.2f$\pm$%.4f' %(CV_MIN_jkk,CV_std_MIN_jkk,NRRP_MIN_jkk,NRRP_std_MIN_jkk))
plt.xlabel('lambda')
plt.ylabel('CV')
for rep in range(N):
    plt.plot(lambda_values, EPSP_CV_dic['%s' %rep], '.')
plt.plot(lambda_values, CV_literature, 'r--', label='literature')
plt.plot(lambda_values, CV_std_literature_up, 'k--', alpha=0.5)
plt.plot(lambda_values, CV_std_literature_down, 'k--', alpha=0.5)
plt.legend()

plt.show()

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






