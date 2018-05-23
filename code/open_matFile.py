import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

RAW_DATA_PATH = '/home/barros/gpfs/bbp.cscs.ch/project/proj35/MVR/data/2011_rodrigo.mat'

mat = scipy.io.loadmat(RAW_DATA_PATH)

#mat is a dictionary,

#print mat['expNames']
#print mat['positions']
#print mat['__header__']
#print mat['__globals__']
#print mat['connections']
#print mat['amp']  #This is an array of arrays, with the amplitudes of the first PSP (I suppose) of each connection
#print mat['__version__']

PSP_am = []
for n in range(len(mat['amp'][0])):
    arr = mat['amp'][0][n]
    for m in range(len(arr)):
        ar = arr[m]
        for a in ar:
            #print a
            if a != 0.0:
                PSP_am.append(a*1000.0)
            else:
                pass

## PSP_am DISTRIBUTION

# mean, var and std
avg_PSP = np.mean(PSP_am)
std_PSP = np.std(PSP_am)
cv_PSP = std_PSP/avg_PSP

var_PSP = np.var(PSP_am)
sigma_PSP = np.sqrt(var_PSP)  # STD
sem_PSP = sigma_PSP / np.sqrt(len(PSP_am))


# We fit the distribution to a gamma function

k_PSP = (avg_PSP*avg_PSP)/(sigma_PSP*sigma_PSP)
Th_PSP = (sigma_PSP*sigma_PSP)/avg_PSP

# From these two values we know the shape of the fitted Gaussian
pdf_x = np.linspace(np.min(PSP_am), np.max(PSP_am), 80)
pdf_y = stats.gamma.pdf(pdf_x, a=k_PSP, scale=Th_PSP)

# figure
plt.figure(1)
plt.hist(PSP_am,20,normed=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('PSP distribution - 500 connections (mean=%.2f; sem=%.2f; std=%.2f)' %(avg_PSP, sem_PSP, sigma_PSP))
plt.xlabel('PSP values')
plt.ylabel('Probability')
plt.grid(True)

print len(PSP_am)

# from this data I will apply a logarithmic function optained in UDF_mean_std.py
# and then get U_se value from that.

U_se = []
A = 0.100153353509
B = 0.433290507093

for f in PSP_am:
    U_val = A*np.log(f) + B
    U_se.append(U_val)

# U_data DISTRIBUTION - NORMAL (Gaussian)

#mean, var and std
avg_U = np.mean(U_se)
var_U = np.var(U_se)
sigma_U = np.sqrt(var_U) #STD
sem_U = sigma_U/np.sqrt(len(U_se))

#print var_U
#print sigma_U

#From these two values we know the shape of the fitted Gaussian
pdf_x = np.linspace(np.min(U_se),np.max(U_se),80)
pdf_y = 1.0/np.sqrt(2*np.pi*var_U)*np.exp(-0.5*(pdf_x-avg_U)**2/var_U)


#figure
plt.figure(2)
plt.hist(U_se,20,normed=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('U_se_log (Rodrigo) norm. distribution (mean=%.2f; sem=%.3f; std=%.2f)' %(avg_U, sem_U, sigma_U))
plt.xticks(np.arange(0,0.9,0.1))
plt.xlabel('U_se values')
plt.ylabel('Probability')
plt.grid(True)

#figure 2 - U_se vs PSP Rodrigo Perin

plt.figure(3)
plt.plot(PSP_am, U_se, 'o')
plt.title('PSP vs U_se Perin\'s data logarithmic')
plt.xlabel('PSP')
plt.ylabel('U_se')

## Now I will try another fitting with a 3 grade polynomial


U_se_2 = []
z1 = 0.0289823281471
z2 = -0.173821618944
z3 = 0.366695867246
z4 = 0.217750189777

for t in PSP_am:
    u = z1*t**3 + z2*t**2 + z3*t + z4
    U_se_2.append(u)

#mean, var and std
avg_U = np.mean(U_se_2)
var_U = np.var(U_se_2)
sigma_U = np.sqrt(var_U) #STD
sem_U = sigma_U/np.sqrt(len(U_se_2))


#From these two values we know the shape of the fitted Gaussian
pdf_x = np.linspace(np.min(U_se_2),np.max(U_se_2),80)
pdf_y = 1.0/np.sqrt(2*np.pi*var_U)*np.exp(-0.5*(pdf_x-avg_U)**2/var_U)

#figure
plt.figure(4)
plt.hist(U_se_2,20,normed=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('U_se_pol (Rodrigo) norm. distribution (mean=%.2f; sem=%.3f; std=%.2f)' %(avg_U, sem_U, sigma_U))
#plt.xticks(np.arange(0,0.9,0.1))
plt.xlabel('U_se values')
plt.ylabel('Probability')
plt.grid(True)

#figure 2 - U_se vs PSP Rodrigo Perin

plt.figure(5)
plt.plot(PSP_am, U_se_2, 'o')
plt.title('PSP vs U_se Perin\'s data polynomial')
plt.xlabel('PSP')
plt.ylabel('U_se')


plt.show()
