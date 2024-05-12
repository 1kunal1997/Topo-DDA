import numpy as np
import math
import matplotlib.pyplot as plt

def parabolic(parameters):
    return parameters - np.square(parameters)

def gradParabolic(parameters):
    return 1 - 2 * parameters

def gaussian(parameters, sig, mu):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((parameters - mu)/sig, 2)/2)/24

def gradGaussian(parameters, sig, mu):
    return -(parameters - mu) * gaussian(parameters, sig, mu) / np.square(sig)

def triangular(parameters, slope):
    result = np.array(parameters)

    for i, para in enumerate(parameters):
        if(para <= 0.5):
            result[i] = slope*para
        else:
            result[i] = slope*(1-para)

    return result

def gradTriangular(parameters, slope):
    result = np.array(parameters)

    for i, para in enumerate(parameters):
        if(para <= 0.5):
            result[i] = slope
        else:
            result[i] = -slope

    return result

def piecewise_update(x, x_max, y_min, y_max):
    if x <= 0.5 * x_max:
        return y_min
    elif 0.5 * x_max < x <= 0.7 * x_max:
        return y_min + (y_max - y_min) / 5
    elif 0.7 * x_max < x <= 0.8 * x_max:
        return y_min + (y_max - y_min) / 2.5
    else:
        return y_max
    
def exp_update(x, x_max, y_min, y_max):
    base = 100
    if x <= 300:
        return y_min + (y_max - y_min) * (math.pow(base, (x / x_max)) - 1) / (base - 1)
    else:
        return 0.5
    
def linear_update(x, x_max, y_min, y_max):
    return y_min + (y_max - y_min) * x / x_max

'''
x_values = np.linspace(0, 1, 200)
plt.figure(2)
plt.plot(x_values, gradGaussian(x_values, 0.1, 0.5))
plt.figure(3)
plt.plot(x_values, gaussian(x_values, 0.1, 0.5))
plt.show()
'''