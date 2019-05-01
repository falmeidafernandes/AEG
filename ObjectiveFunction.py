import numpy as np

"""

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

        fitness = np.sum((ydata-y_pred)**2)
        individual.fitness = fitness

