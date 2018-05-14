import numpy as np
import matplotlib.pyplot as plt
import ou
import corr
from scipy.optimize import curve_fit
from scipy.stats import truncnorm
import time

dt = 0.1 # ms
dt_silico = 0.025
tsim = 400.0 # ms

# v-noise as an OU process

v_sigma = 1.0 # mV

v_mean = -65.0 # mV 

v_tau = 3.0 # ms - correlation time

def reversion(t,v_std_i,tau):
	v_predicted = v_std_i**2*np.exp(-t/tau)
	return v_predicted

def ou_test(iterations,p_initial,v_sigma):
	""" Optimize OU tau for each trial individually """
	tau_store = np.zeros((iterations,),float)
	for i in range(iterations):
		v,t = ou.OU_generator(dt, v_tau, v_sigma, v_mean, tsim)
		v_mean_i,v_std_i = np.mean(v),np.std(v)
		t_corr = np.arange(0,tsim,dt)
		v_corr = corr.autocorr_weave(v-v_mean_i,len(t_corr))
		tau_opt = curve_fit(lambda t_corr, tau: reversion(t_corr,v_std_i,tau),t_corr,v_corr,p0=p_initial)
		tau_store[i] = tau_opt[0]
	return tau_store

def ou_test_vitro(v_raw_traces,p_initial=[5],upper_only=False):
	v_raw_traces = [i-np.mean(i[:950]) for i in v_raw_traces]
	v_raw_traces = [i*1000. for i in v_raw_traces]
	v_raw_traces = [i[6000:10000] for i in v_raw_traces]
	#plt.figure()
	#plt.plot(v_raw_traces)
	tau_opt_store = np.zeros((len(v_raw_traces),),float)
	v_corr_store = []
	for i in range(len(v_raw_traces)):
		v_mean_i,v_std_i = np.mean(v_raw_traces[i]),np.std(v_raw_traces[i])
		#print 'mean voltage', v_mean_i
		#print 'std voltage', v_std_i
		t_corr = np.arange(0,400,dt)
		v_corr = corr.autocorr_weave(v_raw_traces[i]-v_mean_i,len(t_corr))
		#plt.figure()
		#plt.plot(t_corr,v_corr)
		#plt.show()
		tau_opt = curve_fit(lambda t_corr,tau: reversion(t_corr,v_std_i,tau), t_corr, v_corr, p0=p_initial)
		#print 'tau_opt', tau_opt
		tau_opt_store[i] = tau_opt[0]
		v_corr_store.append(v_corr)
	if upper_only is True:
		return np.mean(sorted(tau_opt_store)[int(len(tau_opt_store)*0.67):]),v_corr_store
	else:
		return np.mean(tau_opt_store),v_corr_store

def ou_test_vitro_NAT(v_raw_traces,p_initial=[5],upper_only=False):
	#v_raw_traces = [i-np.mean(i[:950]) for i in v_raw_traces]
	v_raw_traces = [i*1000. for i in v_raw_traces]
	v_raw_traces = [i[6000:10000] for i in v_raw_traces]
	#plt.figure()
	#plt.plot(v_raw_traces)
	#plt.show()
	tau_opt_store = np.zeros((len(v_raw_traces),),float)
	v_corr_store = []
	v_mean = []
	v_std = []
	for i in range(len(v_raw_traces)):
		v_mean_i,v_std_i = np.mean(v_raw_traces[i]),np.std(v_raw_traces[i])
		v_mean.append(v_mean_i)
		v_std.append(v_std_i)
		t_corr = np.arange(0,400,dt)
		v_corr = corr.autocorr_weave(v_raw_traces[i]-v_mean_i,len(t_corr))
		#plt.figure()
		#plt.plot(t_corr, v_corr, 'r')
		#plt.show()
		tau_opt = curve_fit(lambda t_corr,tau: reversion(t_corr,v_std_i,tau), t_corr, v_corr, p0=p_initial)
		tau_opt_store[i] = tau_opt[0]
		v_corr_store.append(v_corr)
	#plt.plot(t_corr, np.transpose(v_corr_store), 'g')
	#plt.show()
	if upper_only is True:
		return np.mean(sorted(tau_opt_store)[int(len(tau_opt_store)*0.67):]),v_corr_store
	else:
		return tau_opt_store,v_corr_store, v_mean, v_std

def ou_test_silico_NAT(v_raw_traces,p_initial=[5],upper_only=False):
	#v_raw_traces = [i-np.mean(i[:3800]) for i in v_raw_traces]
	#v_raw_traces = [i*1000. for i in v_raw_traces]
	#v_raw_traces = [i[200:4000] for i in v_raw_traces]
	v_raw_traces = [i[24000:40000] for i in v_raw_traces]
	tau_opt_store = np.zeros((len(v_raw_traces),),float)
	v_corr_store = []
	v_mean = []
	v_std = []
	for i in range(len(v_raw_traces)):
		v_mean_i,v_std_i = np.mean(v_raw_traces[i]),np.std(v_raw_traces[i])
		v_mean.append(v_mean_i)
		v_std.append(v_std_i)
		t_corr = np.arange(0,400,dt_silico)
		v_corr = corr.autocorr_weave(v_raw_traces[i]-v_mean_i,len(t_corr))
		tau_opt = curve_fit(lambda t_corr,tau: reversion(t_corr,v_std_i,tau), t_corr, v_corr, p0=p_initial)
		tau_opt_store[i] = tau_opt[0]
		v_corr_store.append(v_corr)
	if upper_only is True:
		return np.mean(sorted(tau_opt_store)[int(len(tau_opt_store)*0.67):]),v_corr_store
	else:
		return tau_opt_store, v_corr_store, v_mean, v_std



def horizontal_stack(cnxns):
	""" Run 30 trials and stack 400 msec of "quiet" baseline voltage information. Optimize OU tau with this big long train """
	tau_store = np.zeros((cnxns,),float)
	p_initial = [5]
	for cnxn in range(cnxns):
		stack = []
		for i in range(30):
			a,b = -v_sigma/0.25,1.0/0.25
			v_std_trial = truncnorm.rvs(a,b,loc=v_sigma,scale=0.25,size=1)
			v,t = ou.OU_generator(dt,v_tau,v_std_trial,v_mean,tsim)
			for j in v:
				stack.append(j)
		v_mean_stack,v_std_stack = np.mean(stack),np.std(stack)
		t_corr = np.arange(0,400.,dt)
		v_corr = corr.autocorr_weave(stack-v_mean_stack,len(t_corr))
		tau_opt = curve_fit(lambda t_corr, tau: reversion(t_corr,v_std_stack,tau),t_corr,v_corr,p0=p_initial)
		tau_store[cnxn] = tau_opt[0]
	return tau_store
