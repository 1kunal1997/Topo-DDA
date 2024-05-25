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

def _saveCurrentStructure(all_parameters, path, iteration):
    curr_path = os.path.join(path, f"Structure_Values/Structure{iteration}.txt")
    np.savetxt(curr_path, all_parameters, delimiter='\n')

def _saveObjective(all_obj, path):
    curr_path = os.path.join(path, "Obj_Values.txt")
    np.savetxt(curr_path, all_obj, delimiter='\n')

def _savePenaltyCoefficients(all_coeffs, path):
    curr_path = os.path.join(path, "Coeff_Values.txt")
    np.savetxt(curr_path, all_coeffs, delimiter='\n')

def _saveAveragePenalty(avg_penalties, path):
    curr_path = os.path.join(path, "Average_Penalties_Values.txt")
    np.savetxt(curr_path, avg_penalties, delimiter='\n')

def _savePenaltyShape(penalty_type, path):
    curr_path = os.path.join(path, "Penalty_Shape.txt")
    x_values = np.linspace(0, 1, 200)
    if penalty_type == 'parabolic':
        table = np.column_stack((x_values, binarizers.parabolic(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')
        curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
        table = np.column_stack((x_values, binarizers.gradParabolic(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')
    elif penalty_type == 'gaussian':
        table = np.column_stack((x_values, binarizers.gaussian(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')
        curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
        table = np.column_stack((x_values, binarizers.gradGaussian(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')
    elif penalty_type == 'triangular':
        table = np.column_stack((x_values, binarizers.triangular(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')
        curr_path = os.path.join(path, "PenaltyGradient_Shape.txt")
        table = np.column_stack((x_values, binarizers.gradTriangular(x_values, penalty_config)))
        np.savetxt(curr_path, table, delimiter=',')

def _saveCurrentEField(electric_field, path, iteration):
    curr_path = os.path.join(path, f"E-Field_Values/E-Field{iteration}.txt")
    np.savetxt(curr_path, electric_field, delimiter='\n')

def _saveAllPenalties(penalty, path, iteration):
    curr_path = os.path.join(path, f"Penalty_Values/Penalty{iteration}.txt")
    np.savetxt(curr_path, penalty, delimiter='\n')

def _saveAllParams(params, path, iteration):
    curr_path = os.path.join(path, f"Parameter_Values/Param{iteration}.txt")
    np.savetxt(curr_path, params, delimiter='\n')

def _savePenaltyGradients(penalty_gradients, path, iteration):
    curr_path = os.path.join(path, f"Gradient_Penalty_Values/Gradient{iteration}.txt")
    np.savetxt(curr_path, penalty_gradients, delimiter='\n')

def _createDirectories(path):
    save_dict = {
            'Structure_Values': save_structures,
            'Parameter_Values': save_params,
            'Penalty_Values': save_penalties,
            'Gradient_Penalty_Values': save_gradients_penalty,
            'E-Field_Values': save_fields,
        }

    for directory, flag in save_dict.items():
        if flag:
            Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

def _calculatePenaltyGradients(parameters, penalty_type):

    if penalty_type == 'parabolic':
        return binarizers.gradParabolic(parameters, penalty_config)
    elif penalty_type == 'gaussian':
        return binarizers.gradGaussian(parameters, penalty_config)
    elif penalty_type == 'triangular':
        return binarizers.gradTriangular(parameters, penalty_config)
    
def _calculatePenalty(parameters, penalty_type):

    if penalty_type == 'parabolic':
        return binarizers.parabolic(parameters, penalty_config)
    elif penalty_type == 'gaussian':
        return binarizers.gaussian(parameters, penalty_config)
    elif penalty_type == 'triangular':
        return binarizers.triangular(parameters, penalty_config)

def _calculatePenaltyCoefficient(iteration, coeff_type):

    if coeff_type == 'piecewise':
        return binarizers.piecewise_update(iteration, coeff_config)
    if coeff_type == 'piecewise_absolute':
        return binarizers.piecewise_update_absolute(iteration, coeff_config)
    elif coeff_type == 'linear':
        return binarizers.linear_update(iteration, coeff_config)
    elif coeff_type == 'exp':
        return binarizers.exp_update(iteration, coeff_config)
    elif coeff_type == 'constant':
        return coeff_config["coeff_const"]

# CODE STARTS HERE!!!
print(f"the argument passed into calcultion script is: {sys.argv[1]} ")
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

# flags for saving different values from calculation
save_objective = parsed_json["save_objective"]
save_penalty_coefficients = parsed_json["save_penalty_coefficients"]
save_penalty_shape = parsed_json["save_penalty_shape"]
save_average_penalty = parsed_json["save_average_penalty"]
save_structures = parsed_json["save_structures"]
save_fields = parsed_json["save_fields"]
save_gradients_penalty = parsed_json["save_gradients_penalty"]
save_params = parsed_json["save_params"]
save_penalties = parsed_json["save_penalties"]
base_path = parsed_json["base_path"]

#stepArray = np.logspace(-2, 0, 20)
#stepArray = np.round(stepArray, 3)
#print(stepArray)

#penaltyList = ['parabolic', 'gaussian', 'triangular']
penalty_type = parsed_json["penalty_type"]
#coeffList = ['constant', 'linear', 'exp', 'piecewise', 'piecewise_absolute]
coeff_type = parsed_json["coeff_type"]

step_size = parsed_json["step_size"]
evo_max_iter = parsed_json["evo_max_iteration"]

penalty_config = parsed_json["penalty_configs"][penalty_type]
coeff_config = parsed_json["coeff_configs"][coeff_type]
SLOPE = penalty_config["slope"]
        
full_path = base_path + f"TestPenaltyConfig_HalfCylinder_it{evo_max_iter}_eps{step_size}_{penalty_type}Penalty_slope{SLOPE}_{coeff_type}Coeff"
print("Saving value to path: " + full_path)
_createDirectories(full_path)

model = _constructModel()
optimizer = optimizers.AdamOptimizer()
# step_size = 0.01 # This works better
all_objective_values = []
all_coefficients = []
avg_penalties = []

# main iteration loop for gradient descent optimization
for iteration in range(evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")

    objective_value = model.objective()
    print("Objective Value is: " + str(objective_value))
    all_objective_values.append(objective_value) 

    obj_gradients = model.gradients(objective_value)
    
    if save_structures:
        all_parameters = model.allParameters()
        _saveCurrentStructure(all_parameters, full_path, iteration)
    if save_fields:
        electric_field = model.getElectricField()
        _saveCurrentEField(electric_field, full_path, iteration)

    params = model.parameters
    if save_params:
        _saveAllParams(params, full_path, iteration)           #temporary function for debugging gaussian
    penalty_gradients = _calculatePenaltyGradients(params, penalty_type)
    if save_gradients_penalty:
        _savePenaltyGradients(penalty_gradients, full_path, iteration) #also temporary but could keep.
    penalty = _calculatePenalty(params, penalty_type)
    if save_penalties:
        _saveAllPenalties(penalty, full_path, iteration)        #temporary function for debugging gaussian
    avg_penalties.append(np.average(penalty))
    coeff = _calculatePenaltyCoefficient(iteration, coeff_type)
    #coeff = 0.1
    all_coefficients.append(coeff)

    gradients = obj_gradients - coeff*penalty_gradients
    gradients_final = optimizer(gradients)

    step = step_size * gradients_final
    new_params = np.clip(params + step, 0, 1)
    model.parameters = new_params

if save_objective:
    _saveObjective(all_objective_values, full_path)
if save_penalty_coefficients:
    _savePenaltyCoefficients(all_coefficients, full_path)
if save_average_penalty:
    _saveAveragePenalty(avg_penalties, full_path)
if save_penalty_shape:
    _savePenaltyShape(penalty_type, full_path)

parsed_json["full_path"] = full_path

with open(json_file, "w", encoding='utf-8') as jsonFile:
    json.dump(parsed_json, jsonFile, ensure_ascii=False, indent=4)

copied_json_file = os.path.join(full_path, 'config.json')
os.popen(f'copy {json_file} {copied_json_file}')
    



