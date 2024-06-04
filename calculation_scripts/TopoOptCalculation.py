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

def _constructModel():

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

def _savePenaltyCoefficients(all_coeffs, path):
    curr_path = os.path.join(path, "Coeffs.txt")
    np.savetxt(curr_path, all_coeffs, delimiter='\n')

def _saveCurrentStructure(all_parameters, path, iteration):
    curr_path = os.path.join(path, "Structures", f"Structure{iteration}.npy")
    np.save(curr_path, all_parameters)

def _saveCurrentEField(electric_field, path, iteration):
    curr_path = os.path.join(path, f"E-Fields", f"E-Field{iteration}.npy")
    np.save(curr_path, electric_field)

def _saveAllParams(params, path, iteration):
    curr_path = os.path.join(path, "Parameters", f"Param{iteration}.npy")
    np.save(curr_path, params)

def _createDirectories(path):
    directories = ['Structures', 'Parameters', 'Penalties', 'Gradient_Penalties', 'E-Fields']
    for directory in directories:
        Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

def _calculatePenaltyCoefficient(iteration, coeff_type):
    match coeff_type:
        case 'piecewise':
            return binarizers.piecewise_update(iteration, coeff_config)
        case 'piecewise_absolute':
            return binarizers.piecewise_update_absolute(iteration, coeff_config)
        case 'linear':
            return binarizers.linear_update(iteration, coeff_config)
        case 'exp':
            return binarizers.exp_update(iteration, coeff_config)
        case 'constant':
            return coeff_config["coeff_const"]

def _saveStepSizes(all_step_sizes, path):
    curr_path = os.path.join(path, "stepsizes.txt")
    np.savetxt(curr_path, all_step_sizes, delimiter='\n')

# CODE STARTS HERE!!!
json_file = sys.argv[1]
with open(json_file) as user_file:
    parsed_json = json.load(user_file)

sym_axis = parsed_json["sym_axis"]
geometry_shape = parsed_json["geometry_shape"]
pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
wavelength = parsed_json["wavelength"]
initialization = np.loadtxt("initializations/halfcylinder.txt")
#initialization += np.random.uniform(0, 10e-3, size=initialization.shape)
dielectric_constants = [1.01 + 0j, 5.96282 + 3.80423e-7j]
base_path = parsed_json["base_path"]

#stepArray = np.logspace(-2, 0, 20)
#stepArray = np.round(stepArray, 3)
#print(stepArray)

#coeffList = ['constant', 'linear', 'exp', 'piecewise', 'piecewise_absolute']
coeff_type = parsed_json["coeff_type"]

step_size = parsed_json["step_size"]
filter_size = parsed_json["filter_size"]
filter_radii = parsed_json["filter_radii"]
evo_max_iter = parsed_json["evo_max_iteration"]
threshold_iter = parsed_json["threshold_iteration"]
filter_iter = parsed_json["filter_iteration"]
coeff_config = parsed_json["coeff_configs"][coeff_type]
        
full_path = base_path + f"FilterFirst_HalfCylinder_it{evo_max_iter}_thres{threshold_iter}_eps{step_size}_filter{filter_size}"
print("Saving value to path: " + full_path)
data_path = os.path.join(full_path, "Data")
_createDirectories(data_path)

model = _constructModel()
optimizer = optimizers.AdamOptimizer()
# step_size = 0.01 This works better
all_objective_values = []
all_coefficients = []
all_step_sizes = []

# main iteration loop for gradient descent optimization
for iteration in range(evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")

    objective_value = model.objective()
    print("Objective Value is: " + str(objective_value))
    all_objective_values.append(objective_value) 
    obj_gradients = model.gradients(objective_value)

    # These are only used for plotting, can be reproduced from 'parameters'
    all_parameters = model.allParameters()
    _saveCurrentStructure(all_parameters, data_path, iteration)
    electric_field = model.getElectricField()
    _saveCurrentEField(electric_field, data_path, iteration)

    parameters = model.parameters
    _saveAllParams(parameters, data_path, iteration)  

    gradients = obj_gradients
    gradients_final = optimizer(gradients)
    
    # uncomment if you want a variable step size
    '''
    if (iteration+1) % 100 == 0:
        step_size /= 2
    '''
    all_step_sizes.append(step_size)
    step = step_size * gradients_final
    updated_parameters = np.clip(parameters + step, 0, 1)

    if (iteration + 1) % threshold_iter == 0:
        filter = np.ones([filter_size, filter_size], dtype=float)
        filter /= np.sum(filter)
        # Apply the filter using convolve2d.
        filtered_parameters = convolve2d(updated_parameters, filter, mode="same", boundary="symm")
        thresholded_parameters = np.round(filtered_parameters)
        updated_parameters = thresholded_parameters

    model.parameters = updated_parameters

_saveObjective(all_objective_values, data_path)
_saveStepSizes(all_step_sizes, data_path)

parsed_json["full_path"] = full_path

with open(json_file, "w", encoding='utf-8') as jsonFile:
    json.dump(parsed_json, jsonFile, ensure_ascii=False, indent=4)

copied_json_file = os.path.join(full_path, 'config.json')
shutil.copyfile(json_file, copied_json_file)
    



