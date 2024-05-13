import dda_model
from dda_model import optimizers
from dda_model import binarizers
import numpy as np
import json
from pathlib import Path
import os

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

def _saveObjectiveFunction(all_obj, path):
    curr_path = os.path.join(path, "Obj_Values.txt")
    np.savetxt(curr_path, all_obj, delimiter='\n')

def _saveCurrentEField(electric_field, path, iteration):
    curr_path = os.path.join(path, f"E-Field_Values/E-Field{iteration}.txt")
    np.savetxt(curr_path, electric_field, delimiter='\n')

def _createDirectories(path, save_structures, save_fields):

    Path(path).mkdir(parents=True, exist_ok=True)
    if save_structures:
        Path(os.path.join(path, "Structure_Values")).mkdir(parents=True, exist_ok=True)
    if save_fields:
        Path(os.path.join(path, "E-Field_Values")).mkdir(parents=True, exist_ok=True)

def _calculatePenalty(parameters, penalty_type):

    if penalty_type == 'parabolic':
        return binarizers.gradParabolic(parameters)
    elif penalty_type == 'gaussian':
        return binarizers.gradGaussian(parameters, 0.1, 0.5)
    elif penalty_type == 'triangular':
        return binarizers.gradTriangular(parameters, 1)

def _calculatePenaltyCoefficient(iteration, evo_max_iter, coeff_min, coeff_max, coeff_type):

    if coeff_type == 'piecewise':
        return binarizers.piecewise_update(iteration, evo_max_iter, coeff_min, coeff_max)
    elif coeff_type == 'linear':
        return binarizers.linear_update(iteration, evo_max_iter, coeff_min, coeff_max)
    elif coeff_type == 'exp':
        return binarizers.exp_update(iteration, 299, coeff_min, coeff_max)

with open('config.json') as user_file:
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
save_structures = parsed_json["save_structures"]
save_fields = parsed_json["save_fields"]
base_path = parsed_json["base_path"]

#stepArray = np.logspace(-2, 0, 20)
#stepArray = np.round(stepArray, 3)
#print(stepArray)

#penaltyList = ['parabolic', 'gaussian', 'triangular']
penalty_list = parsed_json["penalty_list"]
#coeffList = ['linear', 'exp', 'piecewise']
coeff_list = parsed_json["coeff_list"]

step_size = parsed_json["step_size"]
evo_max_iter = parsed_json["evo_max_iteration"]
coeff_min = parsed_json["coeff_min"]
coeff_max = parsed_json["coeff_max"]

for penalty_type in penalty_list:
    for coeff_type in coeff_list:
        
        new_path = base_path + '_it' + str(evo_max_iter) + '_eps' + str(step_size) + '_penalty_' + penalty_type + '_coeff' + str(coeff_min) + 'to' + str(coeff_max) + '_' + coeff_type
        print("Saving value to path: " + new_path)
        _createDirectories(new_path, save_structures, save_fields)

        model = _constructModel()
        optimizer = optimizers.AdamOptimizer()
        # step_size = 0.01 # This works better
        all_objective_values = []
        
        # main iteration loop for gradient descent optimization
        for iteration in range(evo_max_iter):
            print(f"---------------------------------------STARTING ITERATION {iteration} ------------------------------------------")

            objective_value = model.objective()
            print("Objective Value is: " + str(objective_value))
            all_objective_values.append(objective_value) 

            obj_gradients = model.gradients(objective_value)
            
            if save_structures:
                all_parameters = model.allParameters()
                _saveCurrentStructure(all_parameters, new_path, iteration)
            if save_fields:
                electric_field = model.getElectricField()
                _saveCurrentEField(electric_field, new_path, iteration)

            params = model.parameters
            penalty_gradients = _calculatePenalty(params, penalty_type)
            coeff = _calculatePenaltyCoefficient(iteration, evo_max_iter, coeff_min, coeff_max, coeff_type)
            #coeff = 0.1

            gradients = obj_gradients - coeff*penalty_gradients
            gradients_final = optimizer(gradients)

            step = step_size * gradients_final
            model.parameters = step
        
        if save_objective:
            _saveObjectiveFunction(all_objective_values, new_path)
    



