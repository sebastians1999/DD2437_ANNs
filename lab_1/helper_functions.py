import numpy as np

def mse(x, y, w):
       e = (w @ x) - y   
       return 0.5 * np.mean(e**2)