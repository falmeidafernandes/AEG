import numpy as np

"""
The objective functions are a function that defines a model to be tested
using a set of parameters previously defined, receives as input a list of
Individuals (which are each one a different set of parameter's values) and a
list of training data, and attribute the 'fitness' of each Individual based on
how good its parameters reflect the training data.
"""


def LineFitting(generation, training_data):

    """
    Objective function designed to fit the parameters 'a' and 'b' of the
    line equation: y = a*x + b

    Parameters
    ----------
    generation: list of Individuals
        A list of Individuals that will have their 'fitness' estimated
    training_data: dict
        data containing a set of inputs and expected outputs for the model
        to be fitted using the genetic algorithm

    Returns
    -------
    sets individual.fitness for each individual in 'generation'
    """

    # prepare the training data
    xdata = np.array(training_data['x'])
    ydata = np.array(training_data['y'])

    # estimate the fitness of each individual
    for individual in generation:
        # prepare parameters
        param_a = individual.genome['a']
        param_b = individual.genome['b']

        # evaluate model
        y_pred = param_a * xdata + param_b

        # evaluate fitness
        fitness = np.sum(-(ydata-y_pred)**2)

        # set fitness
        individual.fitness = fitness


def PolyFitting(generation, training_data):

    """
    Objective function designed to fit the parameters of a polynomial in the
    form y = a0 + a1*x + a2*x**2 + ... + an*x**n

    Parameters
    ----------
    generation: list of Individuals
        A list of Individuals that will have their 'fitness' estimated
    training_data: dict
        data containing a set of inputs and expected outputs for the model
        to be fitted using the genetic algorithm

    Returns
    -------
    sets individual.fitness for each individual in 'generation'
    """

    # prepare the training data
    xdata = np.array(training_data['x'])
    ydata = np.array(training_data['y'])

    # Get the polynomial degree. Note: the -1 takes the parameter a0 into account
    poly_deg = len(generation[0].genome.keys()) - 1


    # estimate the fitness of each individual
    for individual in generation:

        # initiate array that will hold the predicted values
        y_pred = np.zeros(len(ydata))

        # evaluate model. Note: +1 needed to take a0 into account
        for i in range(poly_deg+1):
            a_i = individual.genome['a{}'.format(i)]
            y_pred += a_i * xdata**i

        # evaluate fitness
        fitness = np.sum(-(ydata - y_pred) ** 2)

        # set fitness
        individual.fitness = fitness