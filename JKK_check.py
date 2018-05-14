import matplotlib.pyplot as plt
import numpy


STANDARD_DEVIATION_VALUE = 0.4
DISTRIBUTION_CENTER = 0.5 # mean

# build a distribution with a specific mean and std
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
for i in lst_of_nums:
    new = numpy.delete(lst_of_nums, i, 0)
    #print len(new)
    m = numpy.mean(new)
    means.append(m)

MEAN = numpy.mean(means)
for i in means:
    psv = 29.0*(MEAN - i)**2
    #print psv
    std_dif.append(psv)


#std_scaled = numpy.sqrt(numpy.sum(std_dif)/30.0)
std_scaled = numpy.std(std_dif)
#std = numpy.sqrt(numpy.sum(ps_val))

#print ps_val
print 'MEAN_end', MEAN
print 'STDscaled_end', std_scaled
print 'CV_end', std_scaled/numpy.mean(means)

# plt.hist(lst_of_nums, bins=100)
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.show()