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
    The "Individual" class represents one realization of a set of parameters.

    The value of each parameter is stored in the dictionary self.genome (the
    keys of the dictionary are the names of the parameters). Each Individual
    also belong to a generation and may have ancestors. They also store their
    values of fitness and the rank of their fitness in their own generation.

    Parameters
    ----------
    genome: dict
        The dictionary of parameters and parameter values that defines this
        Individual
    ancestor: Individual or list of Individuals or None
        Can be either the Individual that originated this one, or a list of
        Individuals (in the cases when more Individuals are used to generate
        the genome of this Individual). If the Individual is not generated from
        another Individual, its value is None.
    generation: int
        Indicate the index of the generation to which this Individual belong
    fitness: float
        A value that measures how "fit" the Individual is. In other words, it's
        a value that measures how good this set of parameters is for the
        specific problem one is trying to solve. This is meant to be calculated
        by an ObjectiveFunction defined in ./ObjectiveFunctions.py
    generation_rank: int
        The value that indicates the rank of this Individual when compared to
        other individuals in the same generation.
    """
    
    
    def __init__(self, genome, ancestor = None, generation = 0,
                 fitness = np.nan, generation_rank = np.nan):

        # Setting attributes
        self.genome = genome
        self.generation = generation
        self.fitness = fitness
        self.generation_rank = generation_rank
        self.ancestor = ancestor
    
    
    def reproduce(self, mutation_factor = 0.1, N_children = 1000,
                  partner = None, set_range = None):
        
        # IDEA: rewrite this function allowing for a list of partners

        """
        Used to replicate the set of parameters of this Individual to generate
        a list of Individuals with slightly different parameters

        The amount of difference controlled by the :param:mutation_factor. It
        can also be used to generate new individuals by mixing the set of
        parameters of parameters of this Individual with another one specified
        in :param:partner.

        Parameters
        ----------
        mutation_factor: dict
            Controls the amount of difference between the replicated and the
            original genome. In this implementation, the new parameters are
            sampled from a normal distribution around the original parameter
            value, with the mutation_factor of each parameter as the standard
            deviation.
        N_children: int
            The number of Individuals to be generated
        partner: None or Individual
            If an Individual is given, the value of each parameter to be
            resampled is chosen randomly from self or from this Individual
            for each new Individual generated.

        Returns
        -------
        list of Individuals
            List of Individuals replicated from self (and optionally from
            :param:partner), where each Individual contains a slightly
            different set of parameters.
        """

        # Set the number of parents (only supports 1 or 2)
        parents_N = 1 if partner is None else 2

        # Initialize the list of children that will be filled and returned
        children = []

        # If mutation factor given as int or float, turn it into a dict
        if type(mutation_factor) in (int, float):
            mutation_factor_dict = {}
            for param in self.genome.keys():
                mutation_factor_dict[param] = mutation_factor

            mutation_factor = mutation_factor_dict

        # Generate each child
        for i in range(N_children):

            # Create the dictionary that will contain the set of parameters of this child
            child_genome = {}

            # Sample the value of each parameter
            for param in self.genome.keys():

                # Choose parent that will provide the parameter
                chosen_parent = np.random.choice(parents_N)

                # Get the value of the parameter
                if chosen_parent == 0:
                    param_loc = self.genome[param]
                else:
                    param_loc = partner.genome[param]

                # Sample the new value of the parameter around the selected value
                param_sampled = float(np.random.normal(loc = param_loc,
                                                       scale = mutation_factor[param],
                                                       size = 1))

                # If parameter is sampled outside range, ramdonly resample it inside the range
                if set_range is not None:
                    vmin = set_range[param][0]
                    vmax = set_range[param][1]

                    if param_sampled < vmin or param_sampled > vmax:
                        param_sampled = vmin + (vmax - vmin) * np.random.uniform()

                # Include the parameter and its value in the genome of the generated child
                child_genome[param] = param_sampled

            # Set the child ancestors
            if partner is None:
                child_ancestors = self
            else:
                child_ancestors = [self, partner]

            # Create the child 'Individual' using the sampled values for the parameters
            child = Individual(genome = child_genome,
                               ancestor = child_ancestors)

            # Include the child in the generated list of children
            children.append(child)
        
        return children


class GA(object):
    
    """
    Designed to iterate a genetic algorithm through an objective_function
    considering a given set of parameters and their range

    Parameters
    ----------
    genome: dict
        The keys must be the names of the parameters and the values, a list of
        the range that may be covered by these parameters
    objective_function: func
        The function that will be used to attribute the fitness of each set
        of parameters based on a given training data
    selector: func
        The function that determines which Individuals survive for the next
        iteration based on the fitness values attributed by the
        objective_function
    mutation_factor: float or str
        The value of the mutation factor or the method used to calculate it.
        The supported methods are: 'dynamic'
        For more information, see the documentation of the Individual class

    Methods
    -------

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
        Generates a new list of Individuals randomly sampled inside the given
        range for each parameter (defined in self.genome)

        This was implemented to create the first generation of Individuals and
        also to ensure that the whole parameter space is explored by future
        generations of Individuals, avoiding the convergence to local minima
        instead of a global one.

        Parameters
        ----------
        N_spontaneous: int
            Number of Individuals to be randomly generated

        Notes
        -----
        Each parameter is sampled inside the range defined in
        self.genome[param], which must be a list of 2 elements: the minimum
        and maximum values accepted for the parameter 'param'

        Returns
        -------
        list of Individuals
            the list of randomly generated Individuals
        """

        # Create list of new Individuals to be filled and returned
        new_spontaneous_individuals = []

        # For each new Individual to be created, sample the set of parameters
        for i in range(N_spontaneous):

            # Create dictionary that will correspond to the Individual set of parameters
            genome_i = {}

            # Sample the value of each parameter
            for param in self.genome.keys():
                # Get the minimum and maximum values for this parameter
                vmin = self.genome[param][0]
                vmax = self.genome[param][1]

                # Sample value and include in the genome of the new Individual
                genome_i[param] = vmin + (vmax - vmin) * np.random.uniform()

            # Create new Individual using this genome
            individual = Individual(genome=genome_i)

            # Include the generated individual in the Individuals list to be returned
            new_spontaneous_individuals.append(individual)

        return new_spontaneous_individuals


    def dynamic_mutation_factor(self, survivors, sigma = 3):
        """
        Defines one of the possible methods to determine the mutation_factor,
        for each parameter, used in the replication of a given Individual

        In this case, the mutation_factor corresponds to the average 3*sigma
        deviation calculated from the distribution of each parameter of a
        list of survivors

        Parameters
        ----------
        survivors: list of Individuals
            list of Individuals that will originate the next generation

        Returns
        -------
        dict
            the value calculated for the mutation_factor for each parameter
        """

        # Initialize the dictionary of the mutation_factor
        mutation_factor = {}

        # Calculate the standard deviation of each parameter among the survivors
        for param in self.genome.keys():

            # Create list to receive the value of this parameter for each individual
            survivors_param = []

            for survivor in survivors:
                survivors_param.append(survivor.genome[param])

            # set the mutation factor of this parameter
            mutation_factor[param] = sigma*np.std(survivors_param)

        return mutation_factor


    def get_mutation_factor(self):
        """
        Returns the mutation_factor for the current generation depending on the
        input chosen by the user
        """

        # If mutation_factor is already a dictionary, return it
        if type(self.mutation_factor) is dict:
            return self.mutation_factor

        # If single value, turn into a dictionary repeating the given value for all parameters
        elif type(self.mutation_factor) in (int, float):
            mutation_factor = {}
            for param in self.genome.keys():
                mutation_factor[param] = self.mutation_factor

            self.mutation_factor = mutation_factor
            return self.mutation_factor

        # If mutation factor is set as 'dynamic', return it
        elif self.mutation_factor == 'dynamic':
            return self.dynamic_mutation_factor(survivors=self.survivors)

        elif self.mutation_factor == 'dynamic1sigma':
            return self.dynamic_mutation_factor(survivors=self.survivors, sigma = 1)


    def plot2d(self, param1, param2, center=None):
        """

        """

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

        plt.plot([-100,100],[0,0], color = "#000000")
        plt.plot([0,0],[-100,100], color = "#000000")

        plt.gca().set_title('Generation {0}'.format(self.generation))
        plt.gca().set_xlim((self.genome[param1][0], self.genome[param1][1]))
        plt.gca().set_ylim((self.genome[param2][0], self.genome[param2][1]))
        plt.gca().set_xlabel(param1)
        plt.gca().set_ylabel(param2)
        plt.show()

    def create_new_generation(self, N_spontaneous = 1000, survivors = None,
                              N_children_per_survivor = 1000):

        """
        Creates a new generation of Individuals based on a list of survivors.
        This methods updates self.individuals with the new list of Individuals

        Parameters
        ----------
        N_spontaneous: int
            Number of Individuals randomly generated with no relation with the
            precedent generations (see self.create_spontaneous_generation for
            more information)
        survivors: list of Individuals
            Individuals selected to be replicated for the new generation
        N_children_per_survivor: int
            Number of new Individuals to be sampled from each survivor (see
            Individual.reproduce for more information)

        Returns
        -------
        self.individuals: list of Individuals
            Updates the 'individuals' attribute with the generated list of
            Individuals
        """
        
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
            print("mutation factor {0}".format(mutation_factor))

            for survivor in survivors:
                children = survivor.reproduce(N_children=N_children_per_survivor,
                                              mutation_factor=mutation_factor,
                                              set_range=self.genome)
                new_individuals += children

        # Update individuals attribute
        self.individuals = new_individuals

        # Update generation number
        self.generation += 1


    def iterate(self, training_data, max_generations = 100, fitness_threshold = 0.999, N_spontaneous = 1000,
                plot = None, **kargs):
        """
        Iterates the genetic algorithm through a number of generations until
        convergence or until the maximum number of generations is reached

        Parameters
        ----------
        training_data: dict
            data containing a set of inputs and expected outputs for the model
            to be fitted using the genetic algorithm
        max_generations: int
            max number of generations to iterate the genetic algorithm if
            convergence is not reached
        fitness_threshold: float
            the value of fitness for which the model is considered to have
            converged. The chosen value must depend on how the fitness is
            measured by the objective function being used
        N_spontaneous: int
            the number of spontaneous Individuals (sets of parameters) to be
            generated in the first generation and in between two generations.

        Returns
        -------
        self.individuals: list of Individuals
            Updated after each generation
        self.survivors: list of Individuals
            List of survivors of the last iterated generation
        self.overall_fittest: Individual
            The fittest Individual throughout the whole iteration
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

            print('Generation {0} | Overall fittest {1} | fitness {2}'.format(self.generation,
                                                                              self.overall_fittest.genome,
                                                                              self.overall_fittest.fitness))
            # Step 3.5: plot
            if plot == 'plot2d':
                self.plot2d(**kargs)

            # Step 4: check if final is achieve
            if self.overall_fittest.fitness >= fitness_threshold:
                return self.overall_fittest

            elif self.generation == max_generations:
                return self.overall_fittest

            # Step 5: reproduction of new generation
            self.create_new_generation(N_spontaneous = N_spontaneous, survivors = self.survivors)