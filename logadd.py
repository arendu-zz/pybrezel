__author__ = 'arenduchintala'
from math import log, exp


def logadd(x, y):
    """
    trick to add probabilities in logspace
    without underflow
    """
    #Todo: handle special case when x,y=0
    if x == 0.0 and y == 0.0:
        return log(exp(x) + exp(y))
    elif x >= y:
        return x + log(1 + exp(y - x))
    else:
        return y + log(1 + exp(x - y))


def logadd_of_list(a_list):
    logsum = a_list[0]
    for i in a_list[1:]:
        logsum += logadd(logsum, i)
    return logsum


