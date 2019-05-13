import numpy as np

"""
All the possible 'selector' functions used by the genetic algorithm are
defined here.

The selection function receives a list of Individuals, all of which had their
'fitness' already estimated by some 'objective function', and selects the
individuals that will survive for the next generation.
"""

def RankByFittest(generation, N_survivors = 5):
    """
    Select the 'N_survivors' fittest individuals of a given generation

    Parameters
    ----------
    N_survivors: int
        Number of individuals to survive for the next generation
    generation: list of Individuals
        A list of Individuals for which the fitness has already been
        estimated by some 'objective function'

    Returns
    -------
    list of Individuals:
        the list of 'N_survivors' fittest Individuals

    """

    # get list of fitness
    fitness_list = []
    for individual in generation:
        fitness_list.append(individual.fitness)

    # Turn both lists into array to be able to use np.argsort
    fitness_list = np.array(fitness_list)
    generation = np.array(generation)

    # get order of fitness
    order = np.argsort(fitness_list)

    # order the generation according to fitness
    generation_ordered = generation[order][::-1]

    survivors = []

    # Attribute rank to individuals and select survivors
    for rank, individual in zip(range(len(generation_ordered)), generation_ordered):
        individual.generation_rank = rank

        if rank < N_survivors:
            survivors.append(individual)

    return survivors



def BestofQuadrant(generation):
    """
    Selects N_params**2 individuals, where each selected individual is
    the fittest individual in its own quadrant.

    Parameters
    ----------
    generation: list of Individuals
        the list of individuals that will be used to select the best from each
        quadrant

    Returns
    -------
    list of Individuals
        A list containing the fittest individual for each quadrant
    """

    # Get param names
    params = generation[0].genome.keys()

    # Create survivors dict and fill quadrants
    survivors = {}

    # Create list of quadrants
    quadrants = ['']
    for i in range(len(params)):
        new_quadrants = []
        for quadrant in quadrants:
            new_quadrants.append(quadrant+'0')
            new_quadrants.append(quadrant+'1')

        quadrants = new_quadrants

    # Create instance for the survivor of this parameter
    for quadrant in quadrants:
        survivors[quadrant] = None

    # Find the quadrant of each individual and see if it's the best of the quadrant
    for individual in generation:

        # find individual quadrant
        individual_quadrant = ''
        for param in params:
            if individual.genome[param] < 0:
                individual_quadrant += '0'
            else:
                individual_quadrant += '1'

        # Check if this individual is the best of that quadrant so far
        if survivors[individual_quadrant] is None:
            survivors[individual_quadrant] = individual
        else:
            if individual.fitness > survivors[individual_quadrant].fitness:
                survivors[individual_quadrant] = individual

    # Turn survivors dict into survivors array
    survivors = np.array(list(survivors.values()))

    # Order survivors by fitness
    fitness = []
    for survivor in survivors:
        fitness.append(survivor.fitness)

    order = np.argsort(fitness)
    survivors = list(survivors[order][::-1])

    return survivors

