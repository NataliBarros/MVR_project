#import matplotlib.pyplot as plt
import h5py
from ou import OU_generator
#import numpy as np

# WORKING_DIRECTORY = '/home/barros/gpfs/bbp.cscs.ch/project/proj35/MVR/nrrp_fit_full/L5_TTPC:A-L5_TTPC:B/simulations'

# Noise parameters - OLD
#v_tau = 40.2
#v_sigma = 0.12
#v_mean = 0.0

# Noise parameters - calibrated 33 in Vitro traces - 95 ms before first EPSP
v_tau = 8.6
v_sigma = 0.13
v_mean = 0.0

# Noise parameters - calibrated 33 in Vitro traces - between last EPSP and recovery peak
#v_tau = 28.23
#v_sigma = 0.21
#v_mean = 0.0


#DATA_PATH = '/home/barros/Desktop/Project_MVR/MVR_warmupProject/simulation.h5'

def add_noise(data_path):
    """
    This function look for a file (simulation trace) and add noise to each voltage trace of this file
    :param data_path: path to the file to add noise
    :return v_noisy: voltage trace array with the noisy traces
    :return t_exp: time vector
    """
    sim_data = h5py.File(data_path, 'r')
    L = sim_data.keys() # the keys are the seeds number (ex.seed98321869)
    v_noisy = []
    t_exp = []
    #x=0
    for seedstr, grp in sim_data.iteritems():

        # Get data
        t_raw = grp['time'].value
        v_raw = grp['soma'].value #len(v_raw) = 52000

        # Add noise
        dt = t_raw[1] - t_raw[0] #dt=0.025
        dur = t_raw[-1] #dur = 1300.0
        v_noise, t_noise = OU_generator(dt, v_tau, v_sigma, v_mean, dur) # len(v_noise) = 52001
        t = t_raw
        v = v_raw + v_noise[1:] #THIS IS THE NOISY TRACE

        # Store traces
        v_noisy.append(v)
        t_exp.append(t_raw)
        #print len(t_raw) = 52000
        #print len(t_exp[0]) = 52000
        #v_noisy_trans = np.transpose(v)

        #Plot trace without noise and trace with noise
        #plt.figure()
        #plt.plot(t_exp[0],v_noisy_trans, 'r--')
        #plt.plot(t_exp[0],v_raw, 'b-')
        #x = x+1
        #print 'plot %d done' %x
    #plt.show()

    sim_data.close()
    return v_noisy, t_exp



#''' RUN add_noise '''

#v_noisy, t_exp = add_noise(DATA_PATH)

# Write data to HDF5
#data_file = h5py.File('/home/barros/Desktop/Project_MVR/MVR_warmupProject/noise_simulation.h5', 'w')
#data_file.create_dataset('nrrp5', data=v_noisy)
#data_file.close()