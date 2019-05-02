import numpy as np
from matplotlib import pyplot as plt
"""

"""

__author__ = "Felipe Almeida-Fernandes"
__copyright__ = "Copyright 2019, IAG/USP"
__credits__ = ["Felipe Almeida-Fernandes"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Felipe Almeida-Fernandes"
__email__ = "falmeidafernandes@gmail.com"
__status__ = "Development"

# The code must be designed to work like this: (ex. to fit parameters of a line function)
# >>> import aeg
# >>> from aeg.Selector import MostFit
# >>> from aeg.ObjectiveFunction import LineFitting
# >>>
# >>> model = aeg.AEG(genome = {'a': [-10,10], 'b': [-10,10]},
#                     objective_function = LineFitting,
#                     selector = MostFit)
# >>>
# >>> model.iterate(training_data = {'x': x, 'y': y})
# >>> print model.fittest

class Individual(object):
    
    """
    
    """
    
    
    def __init__(self, genome, ancestor = None, generation = 0,
                 fitness = np.nan, generation_rank = np.nan):
        
        """
        
        """
        
        self.genome = genome
        self.generation = generation
        self.fitness = fitness
        self.generation_rank = generation_rank
        self.ancestor = ancestor
    
    
    def reproduce(self, mutation_factor = 0.1, N_children = 1000,
                  partner = None):
        
        """
        
        """
        
        partners_N = 1 if partner is None else 2
        
        children = []

        for i in range(N_children):
            child_genome = {}
            
            for param in self.genome.keys():
                
                choosen_partner = np.random.choice(partners_N)
                
                if choosen_partner == 0:
                    param_loc = self.genome[param]
                else:
                    param_loc = partner.genome[param]
                
                param_sampled = float(np.random.normal(loc = param_loc,
                                                       scale = mutation_factor,
                                                       size = 1))
                
                child_genome[param] = param_sampled

            if partner is None:
                child_ancestors = self
            else:
                child_ancestors = [self, partner]

            child = Individual(genome = child_genome,
                               ancestor = child_ancestors)
            
            children.append(child)
        
        return children


class AEG(object):
    
    """
    
    """
    
    def __init__(self, genome, objective_function, selector, mutation_factor = 0.1):
        
        self.genome = genome
        self.objective_function = objective_function
        self.selector = selector
        self.generation = 0
        self.individuals = []
        self.mutation_factor = mutation_factor

    def order_individuais(self):
        """
        Orders individuals by Q_value rank
        """

    def create_spontaneous_generation(self, N_spontaneous):
        """

        :param N_spontaneous:
        :return:
        """

        new_spontaneous_individuals = []

        for i in range(N_spontaneous):
            genome_i = {}
            for param in self.genome.keys():
                vmin = self.genome[param][0]
                vmax = self.genome[param][1]
                genome_i[param] = vmin + (vmax - vmin) * np.random.uniform()

            individual = Individual(genome=genome_i)

            new_spontaneous_individuals.append(individual)

        return new_spontaneous_individuals


    def dynamic_mutation_factor(self, survivors):
        mutation_factor = 0

        for param in self.genome.keys():
            survivors_param = []

            for survivor in survivors:
                survivors_param.append(survivor.genome[param])

            mutation_factor += np.std(survivors_param)

        mutation_factor = 3*mutation_factor/len(self.genome.keys())

        return mutation_factor


    def get_mutation_factor(self):
        if self.mutation_factor == 'dynamic':
            return self.dynamic_mutation_factor(survivors=self.survivors)
        else:
            return self.mutation_factor


    def plot2d(self, param1, param2, center=None):

        x = []
        y = []

        for individual in self.individuals:
            x.append(individual.genome[param1])
            y.append(individual.genome[param2])

        plt.scatter(x, y, s = 20, color = "#666666", alpha = 0.5, zorder = 0)

        if center is not None:
            plt.scatter(center[0], center[1], s = 200, marker = 'x', color = "#0000AA", alpha = 1, zorder = 1)

        if self.survivors is not None:
            x_survivor = []
            y_survivor = []

            for survivor in self.survivors:
                x_survivor.append(survivor.genome[param1])
                y_survivor.append(survivor.genome[param2])

        plt.scatter(x_survivor, y_survivor, s = 50, color = "#aa0000", alpha = 1, zorder = 0)

        plt.gca().set_title('Generation {0}'.format(self.generation))
        plt.gca().set_xlim((self.genome[param1][0], self.genome[param1][1]))
        plt.gca().set_ylim((self.genome[param2][0], self.genome[param2][1]))
        plt.gca().set_xlabel(param1)
        plt.gca().set_ylabel(param2)
        plt.show()

    def create_new_generation(self, N_spontaneous = 1000, survivors = None,
                              N_children_per_survivor = 1000):
        
        # create list of new individuals
        new_individuals = []

        # keep the survivors in the new generation
        if survivors is not None:
            new_individuals += survivors
        
        # generate spontaneous individuals
        new_spontaneous_individuals = self.create_spontaneous_generation(N_spontaneous=N_spontaneous)
        new_individuals += new_spontaneous_individuals
        
        # survivor's reproduction
        if survivors is not None:
            mutation_factor = self.get_mutation_factor()
            print "mutation factor {0}".format(mutation_factor)

            for survivor in survivors:
                children = survivor.reproduce(N_children=N_children_per_survivor, mutation_factor=mutation_factor)
                new_individuals += children

        # Update individuals attribute
        self.individuals = new_individuals

        # Update generation number
        self.generation += 1


    def iterate(self, training_data, max_generations = 100, fitness_threshold = 0.999, N_spontaneous = 1000,
                plot = None, **kargs):
        """

        :return:
        """

        # Step 0: generate first spontaneous population
        self.create_new_generation(N_spontaneous = N_spontaneous)

        # Begin the loop
        self.overall_fittest = None
        finished = False

        while not finished:

            # Step 1: measure fitness of the current generation
            self.objective_function(self.individuals, training_data)

            # Step 2: select the fittest individuals
            self.survivors = self.selector(generation = self.individuals)

            # Step 3: check assignment of overall fittest
            if self.overall_fittest is None:
                self.overall_fittest = self.survivors[0]
            elif self.survivors[0].fitness > self.overall_fittest.fitness:
                self.overall_fittest = self.survivors[0]

            print 'Generation {0} | Overall fittest {1} | fitness {2}'.format(self.generation,
                                                                              self.overall_fittest.genome,
                                                                              self.overall_fittest.fitness)
            # Step 3.5: plot
            if plot == 'plot2d':
                self.plot2d(**kargs)

            # Step 4: check if final is achieve
            if self.overall_fittest.fitness >= fitness_threshold:
                self.fittest = self.overall_fittest
                return self.overall_fittest

            if self.generation == max_generations:
                self.fittest = self.overall_fittest
                return self.overall_fittest

            # Step 5: reproduction of new generation
            self.create_new_generation(N_spontaneous = N_spontaneous, survivors = self.survivors)