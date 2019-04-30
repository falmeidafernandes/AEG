import numpy as np

"""

"""


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
    
    
    def __init__(self, genome, ancestors = None, generation = 0, 
                 fitness = np.nan, generation_rank = np.nan):
        
        """
        
        """
        
        self.genome = genome
        self.generation = generation
        self.fitness = fitness
        self.generation_rank = generation_rank
        
        if ancestors is None:
            self.ancestors = [np.nan]*self.generation
        else:
            self.ancestors = ancestors
    
    
    def reproduce(self, mutation_factor = 0.1, N_children = 1000,
                  partner = None):
        
        """
        
        """
        
        partners_N = 1 if partner is None else 2
        
        children = []
        children_ancestors = self.ancestors + [self.generation_rank]
        
        for i in range(N_children):
            child_genome = {}
            
            for param in self.genome.keys():
                
                choosen_partner = np.random.choice(partners_N)
                
                if choosen_partner == 0:
                    param_loc = self.genome[param]
                elif choosen_partner == 1:
                    param_loc = partner.genome[param]
                
                param_sampled = float(np.random.normal(loc = param_loc,
                                                       scale = mutation_factor,
                                                       size = 1))
                
                child_genome[param] = param_sampled
                
            child = Individual(genome = child_genome,
                               ancestors = children_ancestors)
            
            children.append(child)
        
        return children




teste = Individual(genome = {'a': 1, 'b': 2})
children = teste.reproduce()
print children[0].genome



class AEG(object):
    
    """
    
    """
    
    def __init__(self, genome, objective_function, selector):
        
        self.genome = genome
        self.objective_function = objective_function
        self.selector = selector
        self.generation = 0
        self.individuals = []
    
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
        ancestors_spontaneous = [np.nan] * self.generation + [0]

        for i in range(N_spontaneous):
            genome_i = {}
            for param in self.genome.keys():
                vmin = self.genome[param][0]
                vmax = self.genome[param][1]
                genome_i[param] = vmin + (vmax - vmin) * np.random.uniform()

            individual = Individual(genome=genome_i,
                                    ancestors=ancestors_spontaneous)

            new_spontaneous_individuals.append(individual)

        return new_spontaneous_individuals


    def create_new_generation(self, N_spontaneous = 1000, survivors = None,
                              N_children_per_survivor = 1000):
        
        # add new list to self.individuals
        self.individuals.append([])
        
        # generate spontaneous individuals
        new_spontaneous_individuals = self.create_spontaneous_generation(N_spontaneous=N_spontaneous)
        self.individuals[-1] += new_spontaneous_individuals
        
        # survivor's reproduction
        if survivors is not None:
            for survivor in survivors:
                children = survivor.reproduce(N_children=N_children_per_survivor)
                self.individuals[-1] += children

        # Update generation number
        self.generation += 1
    
    def iterate(self, training_data, max_iterations = 100, fitness_threshold = 0.999, N_spontaneous = 1000, **kargs):
        """

        :return:
        """

        # Step 0: generate first spontaneous population
        self.create_new_generation(N_spontaneous = N_spontaneous)

        # Begin the loop
        overall_fittest = None
        finished = False
        iteration = 0

        while not finished:
            iteration += 1

            # Step 1: measure fitness of the current generation
            current_generation_individuals = self.individuals[-1]
            self.objective_function(current_generation_individuals, training_data)

            # Step 2: select the fittest individuals
            current_fittest = self.selector(kargs)

            # Step 3: check assignment of overall fittest
            if overall_fittest is None:
                overall_fittest = current_fittest[0]
            else:
                if current_fittest[0].fitness > overall_fittest.fitness:
                    overall_fittest = current_fittest[0]

            # Step 4: check if final is achieve
            if current_fittest[0].fitness >= fitness_threshold:
                self.fittest = overall_fittest
                return overall_fittest

            elif iteration == max_iterations:
                self.fittest = overall_fittest
                return overall_fittest

            # Step 5: reproduction of new generation
            self.create_new_generation(N_spontaneous = N_spontaneous, survivors = current_fittest)