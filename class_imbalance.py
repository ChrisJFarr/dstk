import numpy as np
import random
import pandas as pd

# TODO Pull new ideas from here
# Vaishali_Ganganwar An_overview_ of_classification_algorithms_for_imbalanced_datasets


# TODO add support for multi class


def classbalancer(x_train, y_train):
    """
    :param x_train: List or series of independent variables
    :param y_train: List or series of categorical dependent variable
    :return: Two data frames with new training data with all classes balanced
    to the original max class size
    """

    x_list = []
    y_list = []
    # Split into x-class number of data frames and put into list
    for c in np.unique(y_train):
        i = y_train == c
        x_list.append(x_train[i].reset_index(drop=True))  # adding to list and resetting index for cleaness
        y_list.append(y_train[i].reset_index(drop=True))

    # Find length of max class
    max_c = np.max([len(series) for series in x_list])

    # loop through each class and resample to match
    for i in range(0, len(x_list)):
        if len(x_list[i]) < max_c:
            loop_index = x_list[i].index
            resample = np.random.choice(loop_index, max_c)
            x_list[i] = x_list[i].loc[resample]
            y_list[i] = y_list[i].loc[resample]
    # combine the list
    x_train = pd.concat(x_list, ignore_index=True).reset_index(drop=True)
    y_train = pd.concat(y_list, ignore_index=True).reset_index(drop=True)

    # Reset indices and return
    return x_train, y_train
