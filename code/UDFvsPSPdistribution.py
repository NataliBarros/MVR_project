# Natali Barros Zulaica
# 5-Dec-2017

#########################################################################################################
# SCRIPT DESCRIPTION:
# This script opens the file called udf_TauEach.txt generated in udf_worup_loop.py
# that compute U, D and F values from deconvolved traces.
# This script also open a txt file with the amplitude values of the deconvolved traces.
# After open the txt files, this script compute the mean, std of U, D, F and PSP
# and also the plots of U, D and F vs first PSP and fit a linear regression.
#
# At the end of this script there are (commented) also different fittings (logarithmic and polinomial)
# for U vs first PSP.
#########################################################################################################

# import modules
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file_path1 = '/home/barros/Desktop/Project_MVR/MVR_warmupProject/Tsodyks_Markram_example/fitting_36traces_TMmodel.txt'
file_path2 = 'amplitudes_smooth_00.txt'

''' Store U, D and F values in separated lists '''
U_data = []
#U_data = [0.45012845183629047, 0.45278622241780142, 0.4819478177371157, 0.23269410327301299, 0.087403712983708748, 0.51554513170616534, 0.1181990631041886, 0.99062346905373344, 0.36115618192938703, 0.46892515822600167, 0.41944280178067173, 0.17758478596729599, 0.49159225130683437, 0.24932014970926952, 0.54723416586005824, 0.92846161053971688, 0.99999999999999944, 0.39574407072746753, 0.31514019377947811, 0.96460887963716224, 0.27524767584337784, 0.45350805112040371, 0.46988366473208465, 0.4441586909609766, 0.35102752647096291, 0.46253448151559273, 0.50273998866696601, 0.28664855329969818, 0.16879799413928509, 0.39288682899129135, 0.42205626959222975, 0.37025454197705598, 0.16888074457168922]

D_data = []
#D_data = [162.40148141122043, 255.59819345508822, 153.29264605136308, 287.95850709094367, 540.69447732371134, 109.64191207986573, 405.16557588836883, 14.209190244328653, 248.92224914333269, 125.56824520536568, 353.55514308947301, 527.02970553137698, 91.571849351781424, 285.05261481812096, 159.1136661040884, 13.792661528312598, 26.263902485429647, 395.70609451349134, 313.82626939585754, 19.163290204122774, 517.37041122439473, 309.20116278681076, 504.06943950194699, 303.07971823646329, 450.24965789546468, 138.0587828454378, 252.4397102048728, 477.72273435053791, 482.04333199648062, 168.26989051781348, 423.08438256031229, 194.68877097886383, 446.68627955222615]

F_data = []
#F_data = [102.35066815695605, 0.0046359149367152241, 76.597515405324913, 23.836895921604317, 47.4862548859033, 53.448229121591226, 18.936475812583396, 1.5306903033718289, 0.06632723845090549, 235.48173271386764, 0.0085876089809255873, 2.7432669247318131e-05, 110.52365438791691, 27.607520348494557, 72.70993999680887, 971.23733436515852, 0.28068642729595084, 12.449312550011474, 15.93356207623825, 5.8167815427990757, 7.4963474334666103e-06, 1.3781735719630079, 0.0022695336073788042, 96.778445468672416, 5.8034020256059193e-08, 78.869756426783837, 109.40124823482789, 1.5155337538708125e-05, 16.741654807390532, 139.7014536823755, 6.0651308420034411e-06, 57.957407175352145, 22.156894282445151]

with open(file_path1) as f:
    line = f.readlines()
    for i in range(len(line)):
        if 'trec = ' in line[i]:
            a = line[i].split(' ')
            D_data.append(float(a[2]))
        if 'tfac = ' in line[i]:
            b = line[i].split(' ')
            F_data.append(float(b[2]))
        if 'use = ' in line[i]:
            c = line[i].split(' ')
            U_data.append(float(c[2]))

print D_data
print F_data
print U_data


# ''' Store PSPs amplitudes and first PSP amplitude in separated lists '''
# first_peak = []
# all_peaks = []
# with open(file_path2) as f:
#     line = f.readlines()
#     for i in range(len(line)):
#         if '=' in line[i]:
#             p = line[i].split(',')
#             fp = p[0].lstrip('amp = [')
#             first_peak.append(float(fp)*1000.0)
#             all_peaks.append(p[0])
#             all_peaks.append(p[1])
#             all_peaks.append(p[2])
#             all_peaks.append(p[3])
#             all_peaks.append(p[4])
#             all_peaks.append(p[5])
#             all_peaks.append(p[6])
#             all_peaks.append(p[7])
#             all_peaks.append(p[8])

''' COMPUTE DISTRIBUTIONS FOR U, D, F AND PSP '''

''' U_data DISTRIBUTION - NORMAL (Gaussian) '''
#mean, var and std
avg_U = np.mean(U_data)
var_U = np.var(U_data)
sigma_U = np.sqrt(var_U) #STD
sem_U = sigma_U/np.sqrt(len(U_data))

print '(U norm dist) -> Uloc=Umean = %s, Uscale=Ustd = %s, Usem = %s' %(avg_U, sigma_U, sem_U)

#From these two values we know the shape of the fitted Gaussian
pdf_x = np.linspace(np.min(U_data),np.max(U_data),80)
pdf_y = 1.0/np.sqrt(2*np.pi*var_U)*np.exp(-0.5*(pdf_x-avg_U)**2/var_U)

#figure
plt.figure(1)
plt.hist(U_data,30,normed=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('U norm. distribution (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_U, sem_U, sigma_U))
plt.xlabel('U values')
plt.ylabel('Probability')
plt.grid(True)


''' D_data DISTRIBUTION - GAMMA '''
#For gamma distribution we need:
#shape: k = mean^2/std^2
#scale: THETA = std^2/mean

#mean, var and std
avg_D = np.mean(D_data)
var_D = np.var(D_data)
sigma_D = np.sqrt(var_D)
sem_D = sigma_D/np.sqrt(len(D_data))
#values for gamma distribution
k_D = (avg_D/sigma_D)**2
Th_D = (sigma_D**2)/avg_D

print '(D gamma dist) -> Dmean = %s, Dstd = %s, Dshape = %s, Dscale = %s' %(avg_D, sigma_D, k_D, Th_D)

#From these two values we know the shape of the fitted Gamma
pdf_x = np.linspace(np.min(D_data),np.max(D_data),80)
pdf_y = stats.gamma.pdf(pdf_x, a=k_D,loc=0.0, scale=Th_D)

# figure
plt.figure(2)
plt.hist(D_data,30,normed=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('D gamm. distribution (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_D, sem_D, sigma_D))
plt.xlabel('D values')
plt.ylabel('Probability')
plt.grid(True)

''' F_data DISTRIBUTION - GAMMA ''' ##### PROBLEMS PLOTTING THE GAMMA FITTING!!!!!!!!!!!!!!!!!!!!
#For gamma distribution we need:
#shape: k = mean^2/std^2
#scale: THETA = std^2/mean

#mean, var and std
avg_F = np.mean(F_data)
var_F = np.var(F_data)
sigma_F = np.sqrt(var_F)
sem_F = sigma_F/np.sqrt(len(F_data))

k_F = (avg_F/sigma_F)**2
Th_F = (sigma_F)**2/avg_F

print '(F gamma dist) -> Fmean = %s, Fstd = %s, Fshape = %s, Fscale = %s' %(avg_F, sigma_F, k_F, Th_F)

#From these two values we know the shape of the fitted Gaussian
#OTHER WAY OF COMPUTING THE GAMMA PARAMETERS: param_F = stats.gamma.fit(F_data)

pdf_x = np.linspace(np.min(F_data),np.max(F_data),80)
pdf_y = stats.gamma.pdf(pdf_x,a=k_F,loc=0.0,scale=Th_F)#(pdf_x, a=k_F, scale=Th_F)

#figure
plt.figure(3)
plt.hist(F_data,30,normed=True)
#plt.plot(pdf_x, pdf_y, 'k--')
plt.title('F gamm. distribution (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_F, sem_F, sigma_F))
plt.xlabel('F values')
plt.ylabel('Probability')
plt.grid(True)
#plt.show()

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


''' PLOT PSP VS U, VS D AND VS F and fit a linear regression '''
# #LINEAR REGRESSIONS
# U_slope, U_intercept, U_r_value, U_p_value, U_std_err = stats.linregress(first_peak, U_data)
# U_line = []
# for x in first_peak:
#     y = U_slope*x + U_intercept
#     U_line.append(y)
#
# D_slope, D_intercept, D_r_value, D_p_value, D_std_err = stats.linregress(first_peak, D_data)
# D_line = []
# for x in first_peak:
#     y = D_slope*x + D_intercept
#     D_line.append(y)
#
# F_slope, F_intercept, F_r_value, F_p_value, F_std_err = stats.linregress(first_peak, F_data)
# F_line = []
# for x in first_peak:
#     y = F_slope*x + F_intercept
#     F_line.append(y)
#
# #PLOT LINEAR REGRESSIONS
# plt.figure(5)
# plt.plot(first_peak, U_data, 'o')
# plt.plot(first_peak, U_line, 'r')
# plt.xlabel('PSP amp First Peak')
# plt.ylabel('U value')
# plt.title('First peak vs U (slope=%.2f; R=%.2f; p=%.3f)' %(U_slope, U_r_value, U_p_value))
#
# plt.figure(6)
# plt.plot(first_peak, D_data, 'o')
# plt.plot(first_peak, D_line, 'r')
# plt.xlabel('PSP amp First Peak')
# plt.ylabel('D value')
# plt.title('First peak vs D (slope=%.2f; R=%.2f; p=%.2f)' %(D_slope, D_r_value, D_p_value))
#
# plt.figure(7)
# plt.plot(first_peak, F_data, 'o')
# plt.plot(first_peak, F_line, 'r')
# plt.xlabel('PSP amp First Peak')
# plt.ylabel('F value')
# plt.title('First peak vs F (slope=%.2f; R=%.2f; p=%.2f)' %(F_slope, F_r_value, F_p_value))


#### FIT FIRST PSP VS U TO LOGARITHMIC FUNCTION ####
#
# A, B = np.polyfit(np.log(first_peak), U_data, 1)
#
# #print A
# #print B
#
# U_log = []
# for f in first_peak:
#     val = A*np.log(f) + B
#     U_log.append(val)
#
#
# plt.figure(8)
# plt.plot(first_peak, U_data, 'o')
# plt.plot(first_peak, U_log, 'or')
# plt.xlabel('PSP amp First Peak')
# plt.ylabel('U value')
# #plt.xticks(np.arange(-0.075,-0.05,0.01))
# plt.title('First peak vs U log fitting: U = %.3f*log(PSP)+%.3f' %(A, B))
#
#
# #Another way using polyfit- instead of a log we try with a polinomial
#
# x = first_peak
# y = U_data
#
# z1, z2, z3, z4 = np.polyfit(x,y,3)
#
# U_data_pol = []
# for t in first_peak:
#     u = z1*t**3 + z2*t**2 + z3*t + z4
#     U_data_pol.append(u)
#
# plt.figure(9)
# plt.plot(first_peak, U_data, 'o')
# plt.plot(first_peak, U_data_pol, 'or')
# plt.xlabel('PSP amp First Peak')
# plt.ylabel('U value')
# plt.title('First peak vs U polynomial(3d) fitting')



plt.show()
