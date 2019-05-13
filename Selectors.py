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
