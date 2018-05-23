# Natali Barros Zulaica
# Feb 2018

# This is a package to analyze the classic TM traces

import numpy as np
import scipy
from scipy import signal
from math import factorial

import matplotlib.pyplot as plt


def amp_rise_lat_firstEPSP(sample_connection, STIM_TIMES, time, t_wind_aft):
    """
    This function selects the rise curve (first EPSP) and find the  5, 20 and 80 % of the amplitude.
    Firstly comptes the amplitud as the minimum - maximum in the interval (first_stimulus, max_peak),
    then computes the percentages and find the times for these respective values.
    Tau_rise = 80%_time - 20%_time
    latency = 5%_time - first_stimulus
    :param sample_connection: array with voltage traces oriented (voltage vs trials)
    :param STIM_TIMES: list with the stimuli times
    :param time: list with time steps in s
    :param t_wind_aft: time window to compute max and min
    :return first_EPSP_amp: list with amplitud values in mV
    :return TAU_rise: list with tau_rise values in ms
    :return latency: list woth latency values in ms
    """
    # compute the mean trace
    sample_mean = np.mean(sample_connection, axis=0)

    # select only the trace part for the first peak
    rise_curve = sample_mean[STIM_TIMES[0]:STIM_TIMES[0]+t_wind_aft]
    rise_curve_time =time[STIM_TIMES[0]:STIM_TIMES[0]+t_wind_aft]

    # find max, min and compute amplitude
    max_value = np.max(sample_mean[STIM_TIMES[0]:STIM_TIMES[0]+t_wind_aft])
    min_value_1 = np.min(sample_mean[STIM_TIMES[0]-50:STIM_TIMES[0]])
    min_value_2 = np.min(sample_mean[STIM_TIMES[0]+10:STIM_TIMES[0]+100]) # for in silico +500
    amp_value = np.abs(max_value-min_value_1)

    first_EPSP_amp = amp_value*1000.0

    # compute percentages
    twenty_peak_value = amp_value*20.0/100.0
    eighty_peak_value = amp_value*80.0/100.0
    five_peak_value = amp_value * 95.0 / 100.0

    twenty_curve_value = -(twenty_peak_value - max_value)
    eighty_curve_value = -(eighty_peak_value - max_value)
    five_curve_value = -(five_peak_value - max_value)

    for x, s in zip(rise_curve, rise_curve_time):
        if (x == min_value_2):
            min_time = s

    n = 0
    m = 0
    l = 0
    for i, j in zip(rise_curve, rise_curve_time):
        if (n == 0) and (i > twenty_curve_value) and (j > min_time):
            twenty_value = i
            twenty_time = j
            n = 1
        if (m == 0) and (i > eighty_curve_value) and (j > min_time):
            eighty_value = i
            eighty_time = j
            m = 1
        if (l == 0) and (i > five_curve_value) and (j > min_time):
            five_value = i
            five_time = j
            l = 1

    TAU_rise = (twenty_time-eighty_time)*1000.0
    latency = (five_time-0.1)*1000.0 # 0.1 should be the time of the presynaptic AP. But I don't have this value so I used first stim time

    return first_EPSP_amp, TAU_rise, latency

def butter_lowpass_filter(data,cutoff,fs,order=5):
    nyq = 0.5 * fs # Nyquist frequency = half of sampling frequency
    normal_cutoff = float(cutoff/nyq) # normal_cutoff = 0.01
    # Design an Nth order digital or analog Butterworth filter and return the filter coefficients in (B,A) form
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.lfilter(b,a,data)
    return y

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
         The Savitzky-Golay filter removes high frequency noise from data.
         It has the advantage of preserving the original shape and
         features of the signal better than other types of filtering
         approaches, such as moving averages techniques.
         This is an archival dump of old wiki content --- see scipy.org for current material.
         Please see http://scipy-cookbook.readthedocs.org/
         (http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay)
         Parameters
         ----------
         y : array_like, shape (N,)
             the values of the time history of the signal.
         window_size : int
             the length of the window. Must be an odd integer number.
         order : int
             the order of the polynomial used in the filtering.
             Must be less then `window_size` - 1.
         deriv: int
             the order of the derivative to compute (default = 0 means only smoothing)
         Returns
         -------
         ys : ndarray, shape (N)
             the smoothed signal (or it's n-th derivative).
         Notes
         -----
         The Savitzky-Golay is a type of low-pass filter, particularly
         suited for smoothing noisy data. The main idea behind this
         approach is to make for each point a least-square fit with a
         polynomial of high order over a odd-sized window centered at
         the point.
         Examples
         --------
         t = np.linspace(-4, 4, 500)
         y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
         ysg = savitzky_golay(y, window_size=31, order=4)
         import matplotlib.pyplot as plt
         plt.plot(t, y, label='Noisy signal')
         plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
         plt.plot(t, ysg, 'r', label='Filtered signal')
         plt.legend()
         plt.show()
         References
         ----------
         .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
            Data by Simplified Least Squares Procedures. Analytical
            Chemistry, 1964, 36 (8), pp 1627-1639.
         .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
            W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
            Cambridge University Press ISBN-13: 9780521880688
         """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def deconvolver(mean_trace, TAU_MEM):
    """
    This function computes the deconvolution of mean trace
    :param mean_trace: data array 1D (list)
    :param TAU_MEM: membrane constant (ms)
    :return deconvolved_trace: list with the voltage values for the deconvolved mean_trace
    """
    x = range(len(mean_trace))
    dx = x[1] - x[0]       #dx==1 -> as this is the distance between points in mean_trace, we have to count the time in 10ths of miliseconds (really dx=0.0001)
    #print "derivative..."
    derived_v = np.gradient(mean_trace,dx)
    deconvolved_trace = []
    # DECONVOLVE
    for j in range(len(mean_trace)):
        top = (TAU_MEM*derived_v[j])+mean_trace[j]
        deconvolved_trace.append(top)
    return deconvolved_trace


def cropping(deconv_trace, STIM_TIMES, t_wind_bef, t_wind_aft):
    """
    This function computes a baseline for the deconvolved trace
    :param deconv_trace:
    :param STIM_TIMES:
    :param t_wind_bef:
    :param t_wind_aft:
    :return:
    """
    crop1 = deconv_trace[0:STIM_TIMES[0]-t_wind_bef]
    crop2 = deconv_trace[STIM_TIMES[7]+t_wind_aft:STIM_TIMES[8]-t_wind_bef]
    totalcrop = crop1.tolist()+crop2.tolist()
    baseline = np.mean(totalcrop)
    return baseline


def compute_amplitude(deconv_trace, STIM_TIMES, t_wind_bef, t_wind_aft):
    """
    This function compute the amplitudes of the EPSPs in deconv_trace
    :param deconv_trace: deconvelved voltage data 1D-array (list)
    :param STIM_TIMES: times where a stimulus is performed
    :param t_wind_bef: time window after the stimulus to find the min value
    :param t_wind_aft: time window after the stimulus to find the max value
    :return amplitudes: list with amplitude values
    """
    amplitudes = []
    max = []
    min = []
    for t in STIM_TIMES:
        #amp = np.max(deconv_trace[t:(t + t_wind_aft)])-np.min(deconv_trace[(t-t_wind_bef):t])
        mx = np.max(deconv_trace[t:(t + t_wind_aft)])
        mn = np.min(deconv_trace[(t - t_wind_bef):t])
        amp = np.abs(mx - mn)
        max.append(mx)#float("{0:.2f}".format(mx)))
        min.append(mn)#float("{0:.2f}".format(mn)))
        amplitudes.append(amp)#(float("{0:.2f}".format(amp)))
    return max, min, amplitudes

def compute_amplitude2(deconv_trace, STIM_TIMES, t_wind_aft):
    """
    This function compute the amplitudes of the EPSPs in deconv_trace
    :param deconv_trace: deconvelved voltage data 1D-array (list)
    :param STIM_TIMES: times where a stimulus is performed
    :param t_wind_aft: time window after the stimulus to find the max value
    :return amplitudes: list with amplitude values
    """
    amplitudes = []
    max = []
    min = []
    for t in STIM_TIMES:
        mx = np.max(deconv_trace[t:(t + t_wind_aft)])
        mn = np.min(deconv_trace[t:(t + t_wind_aft)])
        amp = np.abs(mx - mn)
        max.append(mx)#float("{0:.2f}".format(mx)))
        min.append(mn)#float("{0:.2f}".format(mn)))
        amplitudes.append(amp)#(float("{0:.2f}".format(amp)))
    return max, min, amplitudes


def cv_deconv(sample_connection,STIM_TIMES, t_wind_bef, t_wind_aft, TAU_MEM, R_INPUT, fs):
    """
    This function compute the CV profile of the EPSP amplitudes in sample_connection
    This function filters and deconvolve the sweep before computing the amplitudes
    :param sample_connection: array with voltage traces
    :return amp_cv: list with cv values for each PSP
    """
    amp_array = []
    for trace in sample_connection:
        dec_trace = deconvolver(trace,TAU_MEM)
        data = dec_trace-dec_trace[0]
        dec_filter = butter_lowpass_filter(data, 50.0, fs, order=5)
        max, min, amp = compute_amplitude(dec_filter,STIM_TIMES, t_wind_bef, t_wind_aft)
        amp_array.append(amp)

    ''' compute cv'''
    amp_mean = np.mean(amp_array, axis=0)
    amp_std = np.std(amp_array, axis=0)
    amp_cv = amp_std/amp_mean
    return amp_cv


def cv_JKK(sample_connection, STIM_TIMES, t_wind_aft):
    """ This function computes the Jack Knife (bootstraping) mean traces from a set of traces in sample_connection.
    Also computes the peaks (max values) and minimum of these mean traces and the times for the peaks.
    From the max and min it computes the amplitudes and from amplitudes it computes the CV for each EPSP
    :param sample_connection: data array with all the sweeps of the sample connection
    :param STIM_TIMES: times where a stimulus is performed as time steps (ex: stim1 at 0.1 s; sample_frec = 10KHz -> stim1 = 1000)
    :param t_wind_aft: time window after the stimulus to find the max value (value in time steps)
    :return CV
    """
    # Jackknife bootstraping:
    # make the mean of the traces eliminating each time one, and extract the peaks amplitude for each mean trace
    # the std has to be scaled to (n - 1) as we are resampling in 1/(n - 1) so without scaling the std is very small

    jkk_means = [] # safe EPSP amplitudes from JKK samples
    for sweep in range(len(sample_connection)):
        # remove one different sweep from the trace set in each iteration and compute the mean
        new_sample = np.delete(sample_connection, sweep, 0)
        jkk_sample = np.mean(new_sample, axis=0)

        # compute amplitudes from jkk_sample
        max, min, amplitudes = compute_amplitude2(jkk_sample, STIM_TIMES, t_wind_aft)

        jkk_means.append(amplitudes)

    # compute mean
    amp_MEAN = np.mean(jkk_means, axis=0)

    # compute std - don't use np.std, make the difference
    DIF = []
    for am in jkk_means:
        dif = (am - amp_MEAN)**2
        DIF.append(dif)

    N = np.float(len(sample_connection))
    scale_fact = np.float(len(sample_connection)-1)
    amp_std = scale_fact*np.sqrt(np.sum(DIF, axis = 0)/N)

    # compute CV = std/mean
    CV = amp_std/amp_MEAN

    return CV


#################################### JOHN RANON FUNCTIONS FROM mtj.py ####################
#u_before = lambda(u_se,u_last,fac,delta_t): u_last*np.exp(-delta_t/fac) + u_se*(1 - u_last*np.exp(-delta_t/fac))
x_before = lambda(x_last,dep,delta_t): 1.0 - ((1.0 - x_last)*np.exp(-delta_t/dep))

u_before = lambda(u_se,u_last,fac,delta_t): u_last*(1-u_se)*np.exp(-delta_t/fac) + u_se

DELTA_T_ARRAY = [0.,50.,50.,50.,50.,50.,50.,50.,550.] #msec

def compute_fx(u_se, u_last, fac, x_last, dep, delta_t):
    u_spike = u_before([u_se, u_last, fac, delta_t])
    x_spike = x_before([x_last, dep, delta_t])
    spike_amp = u_spike * x_spike
    x_last = x_spike - x_spike * u_spike
    return [spike_amp, u_spike, x_last]

def spike_amplitudes(u_se, fac, dep):
    u_last = 0.0
    x_last = 1.0
    spike_array = []
    for delta_t in DELTA_T_ARRAY:
        spike_amp, u_last, x_last = compute_fx(u_se, u_last, fac, x_last, dep, delta_t)
        # print 'U_last=%s' % (u_last)
        # print 'x_last=%s' % (x_last)
        spike_array.append(spike_amp)
    return spike_array

def compute_fitness(u_se,fac,dep,experimental):
	spike_array = spike_amplitudes(u_se,fac,dep)
	scaled_spike_array = [i/spike_array[0] for i in spike_array]
	distance = [(i-j)**2 for i,j in zip(scaled_spike_array,experimental)]
	return [np.mean(distance), scaled_spike_array]
