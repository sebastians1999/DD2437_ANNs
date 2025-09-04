import numpy as np

def mse(x, y, w):
       e = (w @ x) - y   
       return 0.5 * np.mean(e**2)

def accuracy(x, y, w):
    e = (w @ x)
    pred = np.where(e>0, 1, 0)
    true = np.where(y>0, 1, 0)
    acc = np.count_nonzero(pred == true) / y.size
    return acc