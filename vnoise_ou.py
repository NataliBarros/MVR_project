# self consistent OU v-noise generator example
# Generate an OU process with mean, std, correlation time
# and demonstrates how to extract those parameters again.

# This method can then be used to parameterize an OU process for neural voltage traces
# 
# Author: Eilif Muller, 2015 

import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import ou
import corr


dt = 0.01 # ms
#tsim = 10000.0 # ms
tsim = 13000.0 # ms

# v-noise as an OU process
# v_sigma = 4.0 # mV
# v_mean = -55.0 # mV
# v_tau = 3.0 # ms - correlation time

# Values from 33 in vitro traces
v_sigma = 0.12 # mV
v_mean = 0.0 # mV
v_tau = 31.5 # ms - correlation time


v,t = ou.OU_generator(dt, v_tau, v_sigma, v_mean, tsim)


# correlations

t_corr = numpy.arange(0,v_tau*10.0,dt)
v_corr = corr.autocorr_weave(v-v_mean,len(t_corr))


# plot

plt.figure()
plt.plot(t_corr,v_corr,'b')
plt.plot(t_corr,v_sigma**2*numpy.exp(-t_corr/v_tau),'k--')
plt.show()

print numpy.mean(v)
print numpy.std(v)


