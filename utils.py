""" Collection of helpful utilities for machine learning
    Author: Zander Blasingame
    Organization: CAMEL at Clarkson University """

import numpy as np
from functools import reduce


def parse_csv(filename, normalize=True):
    headers = []
    data = []

    header_line = True

    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue

            line = line[:-1]  # remove newline from string
            entries = line.split(',')

            if header_line:
                headers = entries
                header_line = False
            else:
                data.append([float(el) for el in entries])

    rtn_mat = np.matrix(data)

    if normalize:
        rtn_mat /= np.abs(rtn_mat).max(axis=1)

    return np.array(headers), rtn_mat
