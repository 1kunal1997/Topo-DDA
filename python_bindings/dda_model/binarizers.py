import numpy as np
import math
import matplotlib.pyplot as plt

def parabolic(parameters):
    return parameters - np.square(parameters)

def gradParabolic(parameters):
    return 1 - 2 * parameters

def gaussian(parameters, sig, mu):
    return sig*np.exp(-np.power((parameters - mu)/sig, 2)/2 + 0.5)

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
    if x <= 200:
        return y_min
    elif 200 < x <= 280:
        return y_min + (y_max - y_min) / 5
    elif 280 < x <= 320:
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


x_values = np.linspace(0, 1, 200)
path = 'E:\\Calculations\\2024May19\\Testing_HalfCylinder_it400_eps0.01_gaussianPenalty_sig0.1_constantCoeff50'
#for i in range(400):

#penalties = np.loadtxt(path + f'\\Penalties\\Penalty{i}.txt')
#parameters = np.loadtxt(path + f'\\Params\\Param{i}.txt')
#penalty_gradients = np.loadtxt(path + f'\\Gradients_Penalty\\Gradient{i}.txt')

plt.figure()
plt.plot(x_values, gradGaussian(x_values, 0.15, 0.5))
#plt.plot(parameters, penalty_gradients, 'o')
#plt.savefig(path + f"\\Debugging\\debugginggradient{i}.png")

plt.figure()
plt.plot(x_values, gaussian(x_values, 0.15, 0.5))
#plt.plot(parameters, penalties, 'o')
#plt.savefig(path + f"\\Debugging\\debugging{i}.png")
plt.show()

#plt.show()
