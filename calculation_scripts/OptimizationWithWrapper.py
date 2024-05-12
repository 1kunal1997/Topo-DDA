import dda_model
from dda_model import optimizers
from dda_model import binarizers
import numpy as np
import json
from pathlib import Path

def construct_model():

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

def saveCurrentStructure(all_parameters, path, iteration):
    with open(path + '\\Structure_Values\\Structure' + str(iteration) + '.txt', 'w') as f:
        for para in all_parameters:
            f.write(f"{para}\n")

def saveObjectiveFunction(all_obj, path):
    with open(path + '\\Obj_Values.txt', 'w') as f:
        for obj in all_obj:
            f.write(f"{obj}\n")

def saveCurrentEField(electric_field, path, iteration):
    with open(path + '\\E-Field_Values\\E-Field' + str(iteration) + '.txt', 'w') as f:
        for field in electric_field:
            f.write(f"{field}\n")

def createDirectories(path, saveStructuresFlag, saveEFieldsFlag):
    Path(path).mkdir(parents=True, exist_ok=True)

    if saveStructuresFlag:
        Path(path + '\\Structure_Values').mkdir(parents=True, exist_ok=True)
    if saveEFieldsFlag:
        Path(path + '\\E-Field_Values').mkdir(parents=True, exist_ok=True)

def calculatePenalty(parameters, penaltyType):
    if penaltyType == 'parabolic':
        return binarizers.gradParabolic(parameters)
    elif penaltyType == 'gaussian':
        return binarizers.gradGaussian(parameters, 0.1, 0.5)
    elif penaltyType == 'triangular':
        return binarizers.gradTriangular(parameters, 1)

def calculatePenaltyCoefficient(iteration, evo_max_iter, coeff_min, coeff_max, coeffType):
    if coeffType == 'piecewise':
        return binarizers.piecewise_update(iteration, evo_max_iter, coeff_min, coeff_max)
    elif coeffType == 'linear':
        return binarizers.linear_update(iteration, evo_max_iter, coeff_min, coeff_max)
    elif coeffType == 'exp':
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
saveObjective = parsed_json["saveObjective"]
saveStructures = parsed_json["saveStructures"]
saveEFields = parsed_json["saveEFields"]
base_path = parsed_json["base_path"]

#stepArray = np.logspace(-2, 0, 20)
#stepArray = np.round(stepArray, 3)
#print(stepArray)

#penaltyList = ['parabolic', 'gaussian', 'triangular']
penaltyList = parsed_json["penaltyList"]
#coeffList = ['linear', 'exp', 'piecewise']
coeffList = parsed_json["coeffList"]

epsilon = parsed_json["epsilon"]
evo_max_iter = parsed_json["evo_max_iteration"]
coeff_min = parsed_json["coeff_min"]
coeff_max = parsed_json["coeff_max"]

for penaltyType in penaltyList:
    for coeffType in coeffList:

        model = construct_model()
        optimizer = optimizers.AdamOptimizer()
        # epsilon = 0.01 # This works better
        all_objective_values = [0] * evo_max_iter

        newpath = base_path + '_it' + str(evo_max_iter) + '_eps' + str(epsilon) + '_penalty_' + penaltyType + '_coeff' + str(coeff_min) + 'to' + str(coeff_max) + '_' + coeffType
        print("Saving value to path: " + newpath)
        createDirectories(newpath, saveStructures, saveEFields)
        
        # main iteration loop for gradient descent optimization
        for iteration in range(evo_max_iter):
            print("---------------------------------------STARTING ITERATION " + str(iteration) + "------------------------------------------")

            objective_value = model.objective()
            print("Objective Value is: " + str(objective_value))
            all_objective_values[iteration] = objective_value  

            objgradients = model.gradients(objective_value)
            
            if saveStructures:
                all_parameters = model.allParameters()
                saveCurrentStructure(all_parameters, newpath, iteration)
            if saveEFields:
                electricField = model.getElectricField()
                saveCurrentEField(electricField, newpath, iteration)

            params = model.parameters
            penaltygradients = np.array(objgradients)
            penaltygradients = calculatePenalty(params, penaltyType)
            coeff = calculatePenaltyCoefficient(iteration, evo_max_iter, coeff_min, coeff_max, coeffType)
            #coeff = 0.1

            gradients = objgradients - coeff*penaltygradients
            gradients_final = optimizer(gradients)

            step = epsilon * gradients_final
            model.parameters = step
        
        if saveObjective:
            saveObjectiveFunction(all_objective_values, newpath)
    



