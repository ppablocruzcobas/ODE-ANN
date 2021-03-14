
from scipy.integrate import solve_ivp
import numpy as np
import time

methods = ['RK23', 'RK45', 'BDF', 'Radau', 'LSODA']

def func(x, y):
    return -y

def analy(x):
    return np.exp(-x)

x = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

for method in methods:
    # t = time.time()
    # result = solve_ivp(func, [0, 1], [1], method=method, t_eval=x)
    # y = result.y[0].reshape(len(result.y[0]), 1)
    # print(y, result.nfev, time.time() - t)
    y = analy(np.array(x))
    print(y.reshape(len(y), 1))
