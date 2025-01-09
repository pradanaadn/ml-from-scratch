import numpy as np


def accuracy(y_true, y_pred):
    
    len_y_true = len(y_true)
    len_y_false = len(y_pred)
    if len_y_true!=len_y_false:
        raise ValueError('Make sure y_true and y_pred have same length')
    acc = np.mean(np.equal(y_true, y_pred))
    return acc