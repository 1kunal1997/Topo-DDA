import numpy as np
import math
from scipy.signal import convolve2d

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
    result = ita * (math.exp(-beta * (1 - parameters / ita)) - (1 - parameters / ita) * math.exp(-beta))
    indices = np.where(parameters > ita)
    result[indices] = (1 - ita) * (1 - math.exp(-beta * (parameters[indices] - ita) / (1 - ita)) + (parameters[indices] - ita) / (1 - ita) * math.exp(-beta)) + ita
    return result

# return mean-filtered parameters using radius 'filter_size'.
def mean_filter(parameters, filter_size):
    filter = np.ones([filter_size, filter_size], dtype=float)
    filter /= np.sum(filter)
    return convolve2d(parameters, filter, mode="same", boundary="symm")
