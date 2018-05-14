# this is an exact copy of John Ranon's script called "deap_mtj.py"

from deap import creator,base,tools
import random
import AnalyzeTrace as at
import numpy as np
#import matplotlib.pyplot as plt

creator.create('FitnessMin',base.Fitness,weights=(-1.0,))
creator.create('Individual',list,fitness=creator.FitnessMin)

EXPERIMENTAL = None

def params(low,up):
    return [random.uniform(a,b) for a,b in zip(low,up)]

IND_SIZE = 3 # --> check
#IND_SIZE = 2

#Definition of parametres intervals for creating the random U, D, F sets
## Lowest values
LOW_INIT = np.zeros(IND_SIZE)

## Highest values for U = 1.0, F = 1000.0 amd D = 2000.0
UP_INIT = [1.0,1000.0,2000.0] # --> check
#UP_INIT = [1.0,2000.0]


### NGEN = number of generations, CXPB = couple conection parametre? and MTPB = mutation parametre?
CXPB,MTPB,NGEN = 0.5,0.4,500

# We will create a new set of 500 UDFs. From them we will compute the possible maximum of the trace that this create
# and after we will compare them with the real peaks that we have in the real mean-trace
toolbox = base.Toolbox()
toolbox.register('attribute',params,LOW_INIT,UP_INIT)
toolbox.register('individual',tools.initIterate,creator.Individual,toolbox.attribute)
toolbox.register('population',tools.initRepeat,list,toolbox.individual)

def evaluate(test):
    fitness_val = at.compute_fitness(test[0],test[1],test[2],EXPERIMENTAL)[0] #u_se,fac,dep #[0] lo he anadido yo
    return fitness_val

#def evaluate(test): # --> check
#	fitness_val = mtj.compute_fitness(test[0],1E-6,test[1],EXPERIMENTAL) #u_se,fac,dep
#	return fitness_val

# Some limit values for the mutation are also needed.
## lowest values
LOW_MUTATE = list(np.zeros(IND_SIZE))

## highest values for U, F and D
UP_MUTATE = [1.0,1000.0,2000.0]
#UP_MUTATE = [1.0,2000.0] # --> check


toolbox.register('mate',tools.cxTwoPoints)
toolbox.register('mutate',tools.mutPolynomialBounded,eta=0.5,low=LOW_MUTATE,up=UP_MUTATE,indpb=0.1)
toolbox.register('select',tools.selTournament,tournsize=3)
toolbox.register('evaluate',evaluate)

# U_trace = []
# F_trace = []
# D_trace = []
def main_ga():
    if len(EXPERIMENTAL) == 0:
        return [0.,0.,0.,0.]
    pop = toolbox.population(n=500)
    fitnesses = map(toolbox.evaluate,pop)
    # start fitting with genetic algorithm
    for ind,fit in zip(pop,fitnesses):
        ind.fitness.values = (fit,) #assign the fitness to the individuals of a population
    for g in range(NGEN):
        offspring = toolbox.select(pop,len(pop)) #select the best inds
        offspring = map(toolbox.clone,offspring)
        #print 'offspring=%s' % offspring
        for child1,child2 in zip(offspring[::2],offspring[1::2]): #mating is random #child1=starting from 0 select every two steps// child2=starting from 1.....
            if random.random() < CXPB:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Now we have a set of ind = (U, D, F) some of them have a non valid fit_value
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

##### DRAW U, D AND F FITTING EVOLUTION ALLONG GENERATIONS ####################################################################################################
####### NATALI ###########   I store the valid values in another list (valid_ind), just to plot the evolution of searching U, D, F during fitting.#############

        # valid_ind = [ind for ind in offspring if ind.fitness.valid]
        #
        # fitness_valid_val = []
        # for ind,fit in zip(valid_ind,fitnesses): #Give me the values of fitness and look for the smallest
        #     ind.fitness.values = (fit,)
        #     fitness_valid_val.append(ind.fitness.values)
        #
        # min_valid_val_fit = min(fitness_valid_val)
        # #print min_valid_val_fit
        #
        # Opt_ind = []
        # for ind, fit in zip(valid_ind, fitnesses):
        #     if fit == min_valid_val_fit:
        #         Opt_ind = ind
        #         break
        # #print Opt_ind
        # U_trace.append(Opt_ind[0])
        # F_trace.append(Opt_ind[1])
        # D_trace.append(Opt_ind[2])

###############################################################################################################################################################

        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):  # Try to fit the values with bad fitting again
            ind.fitness.values = (fit,)
        pop[:] = offspring

        if g%50 == 0:
            print 'Generation: {0}'.format(g)
    fitness_vals = []
    for j in pop:
        fitness_vals.append(j.fitness.values)
        #print 'fitness_vals = %s' % (fitness_vals)
    minfit = min(fitness_vals)
#	return minfit
    #print ' minfit = %s' % (minfit)

##################################################################
   # plt.figure(1)
   # plt.title('U trace')
   # plt.plot(U_trace)
   # plt.figure(2)
   # plt.title('F trace')
   # plt.plot(F_trace)
   # plt.figure(3)
   # plt.title('D trace')
   # plt.plot(D_trace)
   # plt.show()
###################################################################

    for best_fit in range(len(pop)):
        if pop[best_fit].fitness.values[0] == minfit[0]:
            club = list(pop[best_fit])
            club.append(minfit[0])
            #print 'pop = %s' % (pop)
            #print 'fit_value = %s'%(club[3])
            #print 'club = %s' % club #list with the optimal values for U,D,F and Fit_value
            return club


    #print 'club = %s' % (club)
#			return pop[best_fit][0], 0.0, pop[best_fit][1] # --> check

# 1) population fitness is evaluated;
# 2) create a reflist of most fit individuals;
# 3) go through each ind and mate with probability px. if mate, delete previous ind; 4)

