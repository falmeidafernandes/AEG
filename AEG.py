import numpy as np

"""

"""



class Individual(object):
    
    """
    
    """
    
    
    def __init__(self, genome, ancestors = None, generation = 0, 
                 Q_value = np.nan, generation_rank = np.nan):
        
        """
        
        """
        
        self.genome = genome
        self.generation = generation
        self.Q_value = Q_value
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
                
                param_sampled = float(np.random.normal(loc = self.genome[param],
                                                       scale = mutation_factor,
                                                       size = 1))
                
                child_genome[param] = param_sampled
                
            child = Individual(genome = child_genome,
                               ancestors = children_ancestors)
            
            children.append(child)
        
        return children




teste = Individual(genome = {'a': 1, 'b': 2})
children = teste.reproduce()
children[0].genome



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
    
    
