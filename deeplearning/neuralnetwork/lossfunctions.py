# Project: Neural Network From Scratch
# Date: Mars 2019
# Author: Flavien LOISEAU

import numpy as np

def mse(y_targ, y_pred):
    mse = np.mean(np.square(np.subtract(y_pred,y_targ)))
    # print(f"mse: {mse}")
    return mse

def mse_prime(y_targ, y_pred):
    return 2 * np.subtract(y_pred,y_targ) / np.size(y_pred,1)

def cross_entropy(y_targ, y_pred, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = np.size(y_pred,1)
    ce = -np.sum(y_targ*np.log(y_pred))/N
    return ce