import numpy as np

"""

"""



class Individual(object):
    
    """
    
    """
    
    
    def __init__(self, genome, ancestors = None, Q_value = np.nan,
                 generation_rank = np.nan):
        
        """
        """
        
        self.genome = genome
        self.ancestors = ancestors
        self.Q_value = Q_value
        self.generation_rank = generation_rank
    
    
    
    def reproduce(self, mutation_factor = 0.1, N_children = 1000):
        
        children = []
        children_ancestors = self.ancestors + [self.generation_rank]
        
        for i in range(N_children):
            child_genome = {}
            
            for param in self.genome.keys():
            
                child_genome[param] = np.random.normal(loc = self.genome[param],
                                                       scale = mutation_factor,
                                                       size = 1)
                
            child = Individual(genome = child_genome,
                               ancestors = children_ancestors)
            
            children.append(child)
        
        return children






class AEG(object):
    
    """
    """
    
    def __init__(self, genome, Q_function):
        
        self.genome = genome
        self.Q_function = Q_function
        self.generation = 0
        self.individuals = []
    
    def order_individuais(self):
        """
        Orders individuals by Q_value rank
        """
    
    def new_generation(self, N_expontaneous, N_survivors,
                       N_children_per_survivor, expontaneous_genome_range):
        
        # add new list to self.individuals
        self.individuals.append([])
        
        # expontaneously generated
        ancestors_expontaneous = [np.nan]*self.generation + [0]
        
        for i in range(N_expontaneous):
            genome_i = {}
            for param in self.genome.keys():
                vmin = expontaneous_genome_range[param][0]
                vmax = expontaneous_genome_range[param][1]
                genome_i[param] = vmin + (vmax-vmin)*np.random.uniform()
            
            individual = Individual(genome = genome_i,
                                    ancestors = ancestors_expontaneous)
            
            self.individuals[-1].append(individual)
        
        # survivor's reproduction
        if self.generation > 0:
            
            for i in range(N_survivors):
                N = N_children_per_survivor
                children = self.individuals[-2][i].reproduce(N_children = N)
                self.individuals[-1] = self.individuals[-1] + children
    
    
