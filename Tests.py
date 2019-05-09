import numpy as np
import AEG as aeg
from ObjectiveFunction import LineFitting
from Selectors import RankByFittest

########################################################################################################################

def test_Individual_class_initialization():
    individual = aeg.Individual(genome={'a': 1, 'b': 2})
    print(individual.genome)
    print(individual.ancestor)

if __name__ == "__main__":
    run = input("Run test_Individual_class_initialization? (y/N): ")

    if run in ("Y", 'y'):
        test_Individual_class_initialization()

########################################################################################################################

def test_Individual_reproduction():
    individual = aeg.Individual(genome={'a': 1, 'b': 2})
    children = individual.reproduce()
    print(children[0].genome)
    print(children[0].ancestor)
    print(children[1].genome)
    print(children[1].ancestor)
    print(children[200].genome)
    print(children[200].ancestor)

if __name__ == "__main__":
    run = input("Run test_Individual_reproduction? (y/N): ")

    if run in ("Y", 'y'):
        test_Individual_reproduction()

########################################################################################################################

def test_LineFitting():
    individual = aeg.Individual(genome={'a': 1, 'b': 2})
    children = individual.reproduce()

    print(children[0].fitness)
    print(children[1].fitness)
    print(children[2].fitness)

    # Training data
    x = np.arange(-10,10,0.1)
    y = 2*x+3
    training_data = {'x': x, 'y': y}

    LineFitting(generation = children, training_data = training_data)

    print(children[0].genome)
    print(children[0].fitness)
    print(children[1].genome)
    print(children[1].fitness)
    print(children[2].genome)
    print(children[2].fitness)


if __name__ == "__main__":
    run = input("Run test_LineFitting? (y/N): ")

    if run in ("Y", 'y'):
        test_LineFitting()

########################################################################################################################

def test_RankByFittest():
    individual = aeg.Individual(genome={'a': 1, 'b': 2})
    children = individual.reproduce()

    # Training data
    x = np.arange(-10,10,0.1)
    y = 2*x+3
    training_data = {'x': x, 'y': y}

    LineFitting(generation = children, training_data = training_data)
    survivors = RankByFittest(generation = children)

    print(survivors)
    for i in range(len(survivors)):
        print("survivor rank {0} | genome {1} | fitness {2}".format(survivors[i].generation_rank,
                                                                    survivors[i].genome,
                                                                    survivors[i].fitness))

if __name__ == "__main__":
    run = input("Run test_RankByFittest? (y/N): ")

    if run in ("Y", 'y'):
        test_RankByFittest()

########################################################################################################################

def test_GA_iterate():
    # Training data
    x = np.arange(-10,10,0.1)
    y = 2*x+3
    training_data = {'x': x, 'y': y}

    # Genome
    genome = {'a': (-10,10), 'b': (-10,10)}

    ModelFitting = aeg.GA(genome = genome,
                           objective_function=LineFitting,
                           selector=RankByFittest)

    ModelFitting.mutation_factor = 'dynamic'
    best = ModelFitting.iterate(training_data=training_data, max_generations = 100, fitness_threshold=-1e-15,
                                plot='plot2d', param1='a', param2='b', center=[2,3])
    print(best)
    print(best.genome)


if __name__ == "__main__":
    run = input("Run test_AEG_iterate? (y/N): ")

    if run in ("Y", 'y'):
        test_GA_iterate()