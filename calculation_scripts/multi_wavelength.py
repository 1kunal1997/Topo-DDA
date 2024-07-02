import dda_model
from dda_model import optimizers
from dda_model import binarizers
import numpy as np
import json
from pathlib import Path
import os
import shutil
import sys
from scipy.signal import convolve2d
import scipy.interpolate

def _constructAllModels(wavelength_array, weight_array):
    all_models = []
    for wavelength in wavelength_array:
        wavelength_meters = wavelength*1e-9
        diel_ext = diel_ext_re(wavelength_meters) + diel_ext_im(wavelength_meters)*1j
        print(f"Dielectric of external medium at wavelength {wavelength}nm is: {diel_ext}")
        diel_mat = diel_mat_re(wavelength_meters) + diel_mat_im(wavelength_meters)*1j
        print(f"Dielectric of material at wavelength {wavelength}nm is: {diel_mat}")

        dielectric_constants = [diel_ext, diel_mat]
        model = _constructModel(wavelength, dielectric_constants)
        all_models.append(model)

    return all_models

def _constructModel(wavelength, dielectric_constants):

    model = dda_model.DDAModelWrapper(
        geometry_shape,
        pixel_size,
        initialization,
        dielectric_constants=dielectric_constants,
        light_direction=light_direction,
        light_polarization=light_polarization,
        light_wavelength_nm=wavelength,
        symmetry_axes=sym_axis,
        verbose=True,
    )
    return model

def _saveObjective(all_obj, path):
    curr_path = os.path.join(path, "Obj.txt")
    np.savetxt(curr_path, all_obj, delimiter='\n')

def _saveBetas(all_betas, path):
    curr_path = os.path.join(path, "betas.txt")
    np.savetxt(curr_path, all_betas, delimiter='\n')

def _saveFilterRadii(all_radii, path):
    curr_path = os.path.join(path, "filter_radii.txt")
    np.savetxt(curr_path, all_radii, delimiter='\n')

def _saveCurrentStructure(all_parameters, path, iteration):
    curr_path = os.path.join(path, "Structures", f"Structure{iteration}.npy")
    np.save(curr_path, all_parameters)

def _saveCurrentEField(electric_field, path, iteration):
    curr_path = os.path.join(path, f"E-Fields", f"E-Field{iteration}.npy")
    np.save(curr_path, electric_field)

def _saveAllParams(params, path, iteration):
    curr_path = os.path.join(path, "Parameters", f"Param{iteration}.npy")
    np.save(curr_path, params)

def _saveParamsBeforeThreshold(params, path, iteration):
    curr_path = os.path.join(path, "Parameters_Before_Threshold", f"Param{iteration}.npy")
    np.save(curr_path, params)

def _createDirectories(path):
    directories = ['Structures', 'Parameters', 'Parameters_Before_Threshold', 'E-Fields']
    for directory in directories:
        Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

def _calculateBeta(iteration, beta_type):
    match beta_type:
        case 'piecewise':
            return binarizers.piecewise_update(iteration, beta_config)
        case 'piecewise_absolute':
            return binarizers.piecewise_update_absolute(iteration, beta_config)
        case 'linear':
            return binarizers.linear_update(iteration, beta_config)
        case 'exp':
            return binarizers.exp_update(iteration, beta_config)
        case 'constant':
            return beta_config["beta_const"]

def _saveStepSizes(all_step_sizes, path):
    curr_path = os.path.join(path, "stepsizes.txt")
    np.savetxt(curr_path, all_step_sizes, delimiter='\n')

def closed_range(start, stop):
    return range(start, stop + 1)

# CODE STARTS HERE!!!
json_file = sys.argv[1]
with open(json_file) as user_file:
    parsed_json = json.load(user_file)

sym_axis = parsed_json["sym_axis"]
geometry_shape = parsed_json["geometry_shape"]
pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
#initialization = np.loadtxt("initializations/halfcylinder.txt")
initialization = np.loadtxt(parsed_json["init_path"])
#initialization += np.random.uniform(0, 10e-3, size=initialization.shape)

wl, diel_ext_im = np.loadtxt(parsed_json["diel_ext_im_path"], delimiter=' ', unpack=True)
diel_ext_im = scipy.interpolate.interp1d(wl, diel_ext_im)
wl, diel_ext_re = np.loadtxt(parsed_json["diel_ext_re_path"], delimiter=' ', unpack=True)
diel_ext_re = scipy.interpolate.interp1d(wl, diel_ext_re)
wl, diel_mat_im = np.loadtxt(parsed_json["diel_mat_im_path"], delimiter=' ', unpack=True)
diel_mat_im = scipy.interpolate.interp1d(wl, diel_mat_im)
wl, diel_mat_re = np.loadtxt(parsed_json["diel_mat_re_path"], delimiter=' ', unpack=True)
diel_mat_re = scipy.interpolate.interp1d(wl, diel_mat_re)
base_path = parsed_json["base_path"]

#stepArray = np.logspace(-2, 0, 20)
#stepArray = np.round(stepArray, 3)
#print(stepArray)

#betaList = ['constant', 'linear', 'exp', 'piecewise', 'piecewise_absolute']
beta_type = parsed_json["beta_type"]

step_size = parsed_json["step_size"]
filter_radii = parsed_json["filter_radii"]
evo_max_iter = parsed_json["evo_max_iteration"]
threshold_iter = parsed_json["threshold_iteration"]
filter_iter = parsed_json["filter_iteration"]
step_iter = parsed_json["step_iteration"]
beta_config = parsed_json["beta_configs"][beta_type]
ita = parsed_json["ita"]
        
full_path = parsed_json["full_path"]
#full_path = base_path + f"HalfCylinder_it{evo_max_iter}_step{step_size}_stepiter{step_iter}_thresiter{threshold_iter}_beta{beta_type}_0_3_5_10_100_filteriter{filter_iter}_r3"
print("Saving value to path: " + full_path)
data_path = os.path.join(full_path, "Data")
_createDirectories(data_path)
wavelength = parsed_json["wavelength"]
weight_array = [0.1, 0.2, 0.4, 0.2, 0.1]
#weight_array = [1]
wavelength_array = [wavelength-2, wavelength-1, wavelength, wavelength+1, wavelength+2]
#wavelength_array = [wavelength]
center_index = wavelength_array.index(wavelength)
all_models = _constructAllModels(wavelength_array, weight_array)
optimizer = optimizers.AdamOptimizer()
# step_size = 0.01 This works better

# stores the weighted average of the objective values for each structure
avg_objective_values = []
# stores objective values from all wavelengths for each structure
all_objective_values = [[0]*len(wavelength_array)]
all_threshold_betas = []
all_filter_radii = []
all_step_sizes = []
objective_value = 0

# Calculate objective value of initialization and save data as iteration 0

# objective value is now a weighted average over obj values over all wavelengths of interest
for i, weight in enumerate (weight_array):
    curr_objective = all_models[i].objective()
    all_objective_values[0][i] = curr_objective
    objective_value += weight*(curr_objective)

parameters = all_models[center_index].parameters
print(f"Gaussian average objective value of initialization is: {objective_value}")
avg_objective_values.append(objective_value)
_saveCurrentStructure(all_models[center_index].allParameters(), data_path, 0)
_saveCurrentEField(all_models[center_index].getElectricField(), data_path, 0)
_saveAllParams(all_models[center_index].parameters, data_path, 0)

'''
ORDER OF OPERATIONS:
1) calculate gradient of current structure
2) run optimizer on gradients
3) if needed, update step size
4) update parameters by taking step with gradients
5) clip to [0,1]
6) if needed, filter the updated parameters
7) if needed, threshold the updated parameters
8) set model parameters
9) calculate objective value of updated structure
10) save all data
'''

# main iteration loop for gradient descent optimization
for iteration in closed_range(1, evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")

    gradients = np.zeros_like(parameters)
    for i, obj in enumerate (all_objective_values[iteration-1]):
        gradients = gradients + weight_array[i]*all_models[i].gradients(obj)
    gradients_final = optimizer(gradients)
    
    # update step size every 'step_iter' iterations
    if iteration % step_iter == 0:
        step_size /= 2
    all_step_sizes.append(step_size)
    step = step_size * gradients_final
    updated_parameters = parameters + step
    updated_parameters = np.clip(updated_parameters, 0, 1)

    # apply a filter every 'filter_iter' iterations
    filter_size = binarizers.filter_radius_update(iteration, filter_radii)
    all_filter_radii.append(filter_size)
    if iteration % filter_iter == 0:
        updated_parameters = binarizers.mean_filter(updated_parameters, filter_size)

    # apply a threshold every 'threshold_iter' iterations
    beta = _calculateBeta(iteration, beta_type)
    all_threshold_betas.append(beta)
    if iteration % threshold_iter == 0:
        _saveParamsBeforeThreshold(updated_parameters, data_path, iteration)
        updated_parameters = binarizers.smooth_thresholding(updated_parameters, ita, beta)

    for i, model in enumerate(all_models):
        model.parameters = updated_parameters

    objective_value = 0
    objective_values = []
    for i, weight in enumerate (weight_array):
        curr_objective = all_models[i].objective()
        objective_values.append(curr_objective)
        objective_value += weight*(curr_objective)
    all_objective_values.append(objective_values)

    parameters = all_models[center_index].parameters
    print(f"Average objective value of structure {iteration} is: {objective_value}")
    print(f"All objective values of structure {iteration} is: {all_objective_values[iteration]}")

    # save obj, updated structure, E-Field, and updated parameters 
    avg_objective_values.append(objective_value)
    _saveCurrentStructure(all_models[center_index].allParameters(), data_path, iteration)
    _saveCurrentEField(all_models[center_index].getElectricField(), data_path, iteration)
    _saveAllParams(parameters, data_path, iteration) 

_saveObjective(avg_objective_values, data_path)
_saveStepSizes(all_step_sizes, data_path)
_saveBetas(all_threshold_betas, data_path)
_saveFilterRadii(all_filter_radii, data_path)

parsed_json["full_path"] = full_path

with open(json_file, "w", encoding='utf-8') as jsonFile:
    json.dump(parsed_json, jsonFile, ensure_ascii=False, indent=4)

copied_json_file = os.path.join(full_path, 'config.json')
shutil.copyfile(json_file, copied_json_file)
    



