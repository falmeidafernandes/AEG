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

    """

    xdata = np.array(training_data['x'])
    ydata = np.array(training_data['y'])

    for individual in generation:
        param_a = individual.genome['a']
        param_b = individual.genome['b']

        y_pred = param_a * xdata + param_b

        fitness = np.sum(-(ydata-y_pred)**2)
        individual.fitness = fitness

