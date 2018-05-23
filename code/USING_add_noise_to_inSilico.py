# Natalia Barros Zulaica
# Dec 2017
#
# This script opens the h5 file (voltage traces) in a folder and adds noise to the file
# the h5 files contain simulated voltage traces.

import os
from add_noise_to_inSilicoh5 import add_noise
import h5py

path_open = '/home/barros/gpfs/bbp.cscs.ch/project/proj35/MVR/nrrp_fit_full/L5PC-L5PC-oldcirc/simulations/'
path_save = '/home/barros/Desktop/Project_MVR/MVR_warmupProject/h5_data/noise_simulation_new_Noise_400/'

i = 0
for folder in os.listdir(path_open):  # as there are many pairs of gids we use listdir to add the folder name to the path
    i = i+1
    # Open new H5 file for writing
    path_to_save = path_save + 'noise_simulation_new400_%d.h5' %i
    data_file = h5py.File(path_to_save, 'w')
    #for n in range(1, 25, 1):
    n = 1
    NRRP = 'nrrp%d' %n
    # go to the simulation file for adding noise
    file_path = path_open + folder + '/' + NRRP + '/' + 'simulation.h5'
    # add noise
    v_noisy, t_exp = add_noise(file_path)
    # Write the noisy data to the new H5 file
    data_file.create_dataset(NRRP, data=v_noisy)
    data_file.close()
    print 'Noise added to simulation_%d' %i
