import numpy as np

"""

"""

def RankByFittest(generation, N_survivors = 5, reverse = False):
    """

    :param generation: list of individuals with self.fitness already calculated
    :param N_survivors: number of survivors to be returned
    :param reverse: if True, the fittest individual is the one that have the lower value of self.fitness
    :return: list of the N_survivors fittest individuals
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
    if reverse is False:
        generation_ordered = generation[order][::-1]
    else:
        generation_ordered = generation[order]

    survivors = []

    # Attribute rank to individuals and select survivors
    for rank, individual in zip(range(len(generation_ordered)), generation_ordered):
        individual.generation_rank = rank

        if rank < N_survivors:
            survivors.append(individual)

    return survivors
