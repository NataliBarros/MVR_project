import matplotlib.pyplot as plt
import numpy

# first EPSP amp from connection 60
# amp = [0.0026562511920928955, 0.0028000026941299438, 0.0029687508940696716, 0.0031687468290328979, 0.0029750019311904907, 0.0030625015497207642, 0.0034562498331069946, 0.0034625008702278137, 0.0034875050187110901, 0.0028812512755393982, 0.003362506628036499, 0.0033812448382377625, 0.002687498927116394, 0.0034437552094459534, 0.0031874999403953552, 0.0031874999403953552, 0.0036937519907951355, 0.0036124959588050842, 0.0030187517404556274, 0.0032749995589256287, 0.0030687525868415833, 0.0032500028610229492, 0.0029250010848045349, 0.0031999945640563965, 0.0032500028610229492, 0.0033374950289726257, 0.0033187493681907654, 0.003237500786781311, 0.0033375024795532227, 0.0020874962210655212]


#### build a distribution with a specific mean and std
STANDARD_DEVIATION_VALUE = 0.4
DISTRIBUTION_CENTER = 1.5 # mean

def single_num(n):
    # Repeats until a number within the scale is found.
    while 1:
        num = numpy.random.normal(loc=DISTRIBUTION_CENTER, scale=STANDARD_DEVIATION_VALUE)
        if abs(DISTRIBUTION_CENTER-num) <= (STANDARD_DEVIATION_VALUE * n):
            return num

# One standard deviation apart.
lst_of_nums = [single_num(n=1) for _ in xrange(30)]
# print lst_of_nums

print 'MEAN_init', numpy.mean(lst_of_nums)
print 'STD_init', numpy.std(lst_of_nums)
print 'CV_init', numpy.std(lst_of_nums)/numpy.mean(lst_of_nums)

# JKK
means = []
#ps_val = []
std_dif = []
for i in range(len(lst_of_nums)):
    new = numpy.delete(lst_of_nums, i, 0)
    m = numpy.mean(new)
    means.append(m)

MEAN = numpy.mean(means)
for i in means:
    psv = (MEAN - i)**2
    #print psv
    std_dif.append(psv)

std_scaled = numpy.float((len(lst_of_nums)-1))*numpy.sqrt(numpy.sum(std_dif)/numpy.float(len(lst_of_nums)))

#print values from JKK sequence
print 'MEAN_end', MEAN
print 'STDscaled_end', std_scaled
print 'CV_end', std_scaled/MEAN

# plt.hist(lst_of_nums, bins=100)
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.show()