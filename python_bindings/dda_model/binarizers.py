import numpy as np
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def piecewise_update_absolute(iter, info_dict):
    iter1 = info_dict["iter1"] 
    iter2 = info_dict["iter2"] 
    iter3 = info_dict["iter3"]
    iter4 = info_dict["iter4"]
    beta1 = info_dict["beta1"]
    beta2 = info_dict["beta2"] 
    beta3 = info_dict["beta3"]
    beta4 = info_dict["beta4"]
    beta5 = info_dict["beta5"]
    if iter < iter1:
        return beta1
    elif iter1 <= iter < iter2:
        return beta2
    elif iter2 <= iter < iter3:
        return beta3
    elif iter3 <= iter < iter4:
        return beta4
    else:
        return beta5

def piecewise_update(iter, info_dict):
    frac1 = info_dict["frac1"] 
    frac2 = info_dict["frac2"] 
    frac3 = info_dict["frac3"]
    frac4 = info_dict["frac4"]
    iter_end = info_dict["iter_end"]
    beta1 = info_dict["beta1"]
    beta2 = info_dict["beta2"] 
    beta3 = info_dict["beta3"]
    beta4 = info_dict["beta4"]
    beta5 = info_dict["beta5"]
    if iter < frac1*iter_end:
        return beta1
    elif frac1*iter_end <= iter < frac2*iter_end:
        return beta2
    elif frac2*iter_end <= iter < frac3*iter_end:
        return beta3
    elif frac3*iter_end <= iter < frac4*iter_end:
        return beta4
    else:
        return beta5
    
def exp_update(iter, info_dict):
    base = info_dict["base"]
    iter_start = info_dict["iter_start"] 
    iter_end = info_dict["iter_end"]
    beta_min = info_dict["beta_min"]
    beta_max = info_dict["beta_max"] 
    if iter < iter_start:
        return beta_min
    elif iter > iter_end:
        return beta_max
    else:
        return beta_min + (beta_max - beta_min) * (math.pow(base, ((iter-iter_start) / (iter_end-iter_start))) - 1) / (base - 1)
    
def linear_update(iter, info_dict):
    iter_start = info_dict["iter_start"] 
    iter_end = info_dict["iter_end"]
    beta_min = info_dict["beta_min"]
    beta_max = info_dict["beta_max"]
    if iter < iter_start:
        return beta_min
    elif iter > iter_end:
        return beta_max
    else:
        return beta_min + (beta_max - beta_min) * iter / iter_end

def filter_radius_update(iter, info_dict):
    r1 = info_dict["r1"]
    it1 = info_dict["it1"]
    r2 = info_dict["r2"]
    it2 = info_dict["it2"]
    if it1 <= iter < it2:
        return r1
    elif iter >= it2:
        return r2
    else:
        return 1

# return thresholded parameters using smooth Heaviside function.
def smooth_thresholding(parameters, ita, beta):
    result = ita * (np.exp(-beta * (1 - parameters / ita)) - (1 - parameters / ita) * np.exp(-beta))
    indices = np.where(parameters > ita)
    result[indices] = (1 - ita) * (1 - np.exp(-beta * (parameters[indices] - ita) / (1 - ita)) + (parameters[indices] - ita) / (1 - ita) * np.exp(-beta)) + ita
    return result

# return mean-filtered parameters using radius 'filter_size'.
def mean_filter(parameters, filter_size):
    filter = np.ones([filter_size, filter_size], dtype=float)
    filter /= np.sum(filter)
    return convolve2d(parameters, filter, mode="same", boundary="symm")

'''
x_values = np.linspace(0, 1, 200)
plt.figure()
for i in [0, 3, 5, 10, 100]:
    threshold_func = smooth_thresholding(x_values, 0.5, i)
    plt.plot(x_values, threshold_func, label=f'$\\beta$={i}')
plt.legend(loc='lower right')
plt.title('Threshold Function')
plt.xlabel('Parameter')
plt.ylabel('Thresholded Parameter')
plt.show()
'''
