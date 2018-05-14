# Natali Barros Zulaica
# Dec - 2017

# This script compute the fitting of a negative exponential
# for the decay curve of the recovery peak



import numpy as np
from scipy.optimize import curve_fit



def func(x, a, b, c):
    x = x - x.min()
    return a * (1 - np.exp(-b * x)) + c

def compute_TauMem(x, y):
    popt, pcov = curve_fit(func, x, y)
    Tau_mem = (1/popt[1])/100.0 #ms
    return popt, pcov, func, Tau_mem


####### Example --------------------------------------------------------------------------------------------------------------------------------------
# # FIT AN EXPONENTIAL IN THE DECAY PART OF THE RECOVERY PEAK -> TAU_MEM --------------------------------------------------------------------------------------
# last_peak_curve = np.ndarray.tolist(sample_mean[td.STIM_TIMES[8]:td.STIM_TIMES[8]+4000])
# peak_value = np.max(sample_mean[td.STIM_TIMES[8]:td.STIM_TIMES[8]+td.t_wind_bef])
# peak_value_index = last_peak_curve.index(peak_value)
# curve_to_fit = last_peak_curve[peak_value_index:]
# x = np.arange(peak_value_index,len(curve_to_fit)+peak_value_index,1)

#def func(x, a, b, c):
#    x = x - x.min()
#    return a * (1 - np.exp(-b * x)) + c

#popt, pcov = curve_fit(func, x, curve_to_fit)

#Tau_mem = (1/popt[1])/100.0 #ms

#popt, pcov, func, Tau_mem = compute_TauMem(x, curve_to_fit)

#td.TAU_MEM = Tau_mem*400.0


#plt.figure()
#plt.plot(x, curve_to_fit, 'b-')
#plt.plot(x, func(x, *popt), 'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#plt.show()
