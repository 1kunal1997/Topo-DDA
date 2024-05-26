import numpy as np
import math
import matplotlib.pyplot as plt

def parabolic(parameters, info_dict):
    return parameters - np.square(parameters)

def gradParabolic(parameters, info_dict):
    return 1 - 2 * parameters

def gaussian(parameters, info_dict):
    sig = info_dict["sigma"]
    mu = info_dict["mu"]
    return sig*np.exp(-np.power((parameters - mu)/sig, 2)/2 + 0.5)

def gradGaussian(parameters, info_dict):
    sig = info_dict["sigma"]
    mu = info_dict["mu"]
    return -(parameters - mu) * gaussian(parameters, info_dict) / np.square(sig)

def triangular(parameters, info_dict):
    slope = info_dict["slope"]
    result = np.array(parameters)

    for i, para in enumerate(parameters):
        if(para <= 0.5):
            result[i] = slope*para
        else:
            result[i] = slope*(1-para)

    return result

def gradTriangular(parameters, info_dict):
    slope = info_dict["slope"]
    result = np.array(parameters)

    for i, para in enumerate(parameters):
        if(para <= 0.5):
            result[i] = slope
        else:
            result[i] = -slope

    return result

def piecewise_update_absolute(iter, info_dict):
    iter1 = info_dict["iter1"] 
    iter2 = info_dict["iter2"] 
    iter3 = info_dict["iter3"]
    coeff1 = info_dict["coeff1"]
    coeff2 = info_dict["coeff2"] 
    coeff3 = info_dict["coeff3"]
    coeff4 = info_dict["coeff4"]
    if iter <= iter1:
        return coeff1
    elif iter1 < iter <= iter2:
        return coeff2
    elif iter2 < iter <= iter3:
        return coeff3
    else:
        return coeff4

def piecewise_update(iter, info_dict):
    frac1 = info_dict["frac1"] 
    frac2 = info_dict["frac2"] 
    frac3 = info_dict["frac3"]
    iter_end = info_dict["iter_end"]
    coeff1 = info_dict["coeff1"]
    coeff2 = info_dict["coeff2"] 
    coeff3 = info_dict["coeff3"]
    coeff4 = info_dict["coeff4"]
    if iter <= frac1*iter_end:
        return coeff1
    elif frac1*iter_end < iter <= frac2*iter_end:
        return coeff2
    elif frac2*iter_end < iter <= frac3*iter_end:
        return coeff3
    else:
        return coeff4
    
def exp_update(iter, info_dict):
    base = info_dict["base"]
    iter_start = info_dict["iter_start"] 
    iter_end = info_dict["iter_end"]
    coeff_min = info_dict["coeff_min"]
    coeff_max = info_dict["coeff_max"] 
    if iter < iter_start:
        return coeff_min
    elif iter > iter_end:
        return coeff_max
    else:
        return coeff_min + (coeff_max - coeff_min) * (math.pow(base, ((iter-iter_start) / (iter_end-iter_start))) - 1) / (base - 1)
    
def linear_update(iter, info_dict):
    iter_start = info_dict["iter_start"] 
    iter_end = info_dict["iter_end"]
    coeff_min = info_dict["coeff_min"]
    coeff_max = info_dict["coeff_max"]
    if iter < iter_start:
        return coeff_min
    elif iter > iter_end:
        return coeff_max
    else:
        return coeff_min + (coeff_max - coeff_min) * iter / iter_end

'''
x_values = np.linspace(0, 1, 200)
plt.figure()
plt.plot(x_values, gradGaussian(x_values, 0.15, 0.5))
plt.figure()
plt.plot(x_values, gaussian(x_values, 0.15, 0.5))
plt.show()
'''
