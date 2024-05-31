import dda_model
from dda_model import optimizers
from dda_model import binarizers
import numpy as np
import json
from pathlib import Path
import os
import sys

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

def _saveAveragePenalty(avg_penalties, path):
    curr_path = os.path.join(path, "Average_Penalties.txt")
    np.savetxt(curr_path, avg_penalties, delimiter='\n')

def _savePenaltyShape(penalty_type, path):
    curr_path = os.path.join(path, "Penalty_Shape.txt")
    x_values = np.linspace(0, 1, 200)
    match penalty_type:
        case 'parabolic':
            table = np.column_stack((x_values, binarizers.parabolic(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')
            curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
            table = np.column_stack((x_values, binarizers.gradParabolic(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')
        case 'gaussian':
            table = np.column_stack((x_values, binarizers.gaussian(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')
            curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
            table = np.column_stack((x_values, binarizers.gradGaussian(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')
        case 'triangular':
            table = np.column_stack((x_values, binarizers.triangular(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')
            curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
            table = np.column_stack((x_values, binarizers.gradTriangular(x_values, penalty_config)))
            np.savetxt(curr_path, table, delimiter=',')

def _saveCurrentStructure(all_parameters, path, iteration):
    curr_path = os.path.join(path, "Structures", f"Structure{iteration}.txt")
    np.savetxt(curr_path, all_parameters, delimiter='\n')

def _saveCurrentEField(electric_field, path, iteration):
    curr_path = os.path.join(path, f"E-Fields", f"E-Field{iteration}.txt")
    np.savetxt(curr_path, electric_field, delimiter='\n')

def _saveAllPenalties(penalty, path, iteration):
    curr_path = os.path.join(path, "Penalties", f"Penalty{iteration}.txt")
    np.savetxt(curr_path, penalty, delimiter='\n')

def _saveAllParams(params, path, iteration):
    curr_path = os.path.join(path, "Parameters", f"Param{iteration}.txt")
    np.savetxt(curr_path, params, delimiter='\n')

def _savePenaltyGradients(penalty_gradients, path, iteration):
    curr_path = os.path.join(path, "Gradient_Penalties", f"Gradient{iteration}.txt")
    np.savetxt(curr_path, penalty_gradients, delimiter='\n')

def _createDirectories(path):
    directories = ['Structures', 'Parameters', 'Penalties', 'Gradient_Penalties', 'E-Fields']
    for directory in directories:
        Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

def _calculatePenaltyGradients(parameters, penalty_type):
    match penalty_type:
        case 'parabolic':
            return binarizers.gradParabolic(parameters, penalty_config)
        case 'gaussian':
            return binarizers.gradGaussian(parameters, penalty_config)
        case 'triangular':
            return binarizers.gradTriangular(parameters, penalty_config)
    
def _calculatePenalty(parameters, penalty_type):
    match penalty_type:
        case 'parabolic':
            return binarizers.parabolic(parameters, penalty_config)
        case 'gaussian':
            return binarizers.gaussian(parameters, penalty_config)
        case 'triangular':
            return binarizers.triangular(parameters, penalty_config)

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

#penaltyList = ['parabolic', 'gaussian', 'triangular']
penalty_type = parsed_json["penalty_type"]
#coeffList = ['constant', 'linear', 'exp', 'piecewise', 'piecewise_absolute']
coeff_type = parsed_json["coeff_type"]

step_size = parsed_json["step_size"]
evo_max_iter = parsed_json["evo_max_iteration"]

penalty_config = parsed_json["penalty_configs"][penalty_type]
coeff_config = parsed_json["coeff_configs"][coeff_type]
        
full_path = base_path + f"Verify2HalfCylinder_it{evo_max_iter}_eps{step_size}_{penalty_type}Penalty_{coeff_type}Coeff"
print("Saving value to path: " + full_path)
data_path = os.path.join(full_path, "Data")
_createDirectories(data_path)

model = _constructModel()
optimizer = optimizers.AdamOptimizer()
# step_size = 0.01 This works better
all_objective_values = []
all_coefficients = []
avg_penalties = []
all_step_sizes = []

# main iteration loop for gradient descent optimization
for iteration in range(evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")

    objective_value = model.objective()
    print("Objective Value is: " + str(objective_value))
    all_objective_values.append(objective_value) 
    obj_gradients = model.gradients(objective_value)

    # These are only used for plotting, can be reproduced from 'params'
    all_parameters = model.allParameters()
    _saveCurrentStructure(all_parameters, data_path, iteration)
    electric_field = model.getElectricField()
    _saveCurrentEField(electric_field, data_path, iteration)

    params = model.parameters
    _saveAllParams(params, data_path, iteration)  
    '''
    v = get_parameters()
    v.reshape([2, -1], order='???')
    v = np.convolve2d(filter, v)
    set_parameters(v.flatten())
    '''         
    penalty_gradients = _calculatePenaltyGradients(params, penalty_type)
    _savePenaltyGradients(penalty_gradients, data_path, iteration) 
    penalty = _calculatePenalty(params, penalty_type)
    _saveAllPenalties(penalty, data_path, iteration)    
    avg_penalties.append(np.average(penalty))

    coeff = _calculatePenaltyCoefficient(iteration, coeff_type)
    #coeff = 0.1
    all_coefficients.append(coeff)

    gradients = (1-coeff)*obj_gradients - coeff*penalty_gradients
    #gradients = obj_gradients - coeff*penalty_gradients
    gradients_final = optimizer(gradients)
    
    # uncomment if you want a variable step size
    '''
    if (iteration+1) % 100 == 0:
        step_size /= 2
    '''
    all_step_sizes.append(step_size)
    step = step_size * gradients_final
    new_params = np.clip(params + step, 0, 1)
    model.parameters = new_params

_saveObjective(all_objective_values, data_path)
_savePenaltyCoefficients(all_coefficients, data_path)
_saveAveragePenalty(avg_penalties, data_path)
_savePenaltyShape(penalty_type, data_path)
_saveStepSizes(all_step_sizes, data_path)

parsed_json["full_path"] = full_path

with open(json_file, "w", encoding='utf-8') as jsonFile:
    json.dump(parsed_json, jsonFile, ensure_ascii=False, indent=4)

copied_json_file = os.path.join(full_path, 'config.json')
os.popen(f'copy {json_file} {copied_json_file}')
    



