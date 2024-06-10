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
from scipy import ndimage

import plotting


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
        verbose=False,
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
wavelength = parsed_json["wavelength"]
initialization = np.loadtxt(parsed_json["init_path"])
diel_ext = parsed_json["diel_ext"]
diel_mat = parsed_json["diel_mat"]
dielectric_constants = [diel_ext[0] + diel_ext[1]*1j, diel_mat[0], diel_mat[1]*1j]

evo_max_iter = parsed_json["evo_max_iteration"]

base_path = parsed_json["base_path"]
run_name = parsed_json["run_name"]
full_path = os.path.join(base_path, run_name)
print("Saving value to path: " + full_path)
plot_path = os.path.join(full_path, "Plots")
Path(plot_path).mkdir(parents=True, exist_ok=True)



model = _constructModel()
all_objective_values = []

# Calculate objective value of initialization and save data as iteration 0
objective_value = model.objective()
parameters = model.parameters
print(f"Objective value of initialization is: {objective_value}")
all_objective_values.append(objective_value)

'''
Genetic algorithm (template).

X = initialization
for iteration in range(num_iters):
    # This applies random "design steps" to X.
    # (1) dilate, (2) run some gradient ops, (3) flip some pixels, (4) ???
    # Traditionally, (3) flip some pixels w.p. P (P ~ 1%)
    # flip + filter + dilate + whatever.
    # Y is going to be "num_evolutions" size
    Y = evolve(X)
    # Next, find the objective for all of the items in Y, and kill off the 
    # bottom ones. X should contain "num_population" size.
    X = prune(Y)

final_design = argmax(objective(X))
'''

def random_flips(parameters, p):
    # p = flipping probability

    ''' This kinda works but suffers from islands.
    flipped_pixels = np.random.choice([0, 1], size=parameters.shape, p=[1-p, p])
    # Filter / dilate / etc to make this more physically realizable.
    # Now, instead of 0/1 salt and pepper noise, we will make blocks.
    flipped_pixels = ndimage.binary_dilation(flipped_pixels, iterations=2)
    flipped_pixels = np.array(flipped_pixels, dtype=float)
    parameters = parameters + flipped_pixels
    parameters = np.mod(parameters, 2)
    '''
    # Flip pixels, but only 0s that are connected to an edge of a 1.
    outer_flips = np.random.choice([0, 1], size=parameters.shape, p=[1-p, p])
    outer_edges = ndimage.binary_dilation(parameters).astype(float) - parameters
    outer_update = outer_edges * outer_flips
    # Apply the outer update.
    parameters = parameters + outer_update
    parameters = np.mod(parameters, 2)

    # Flip pixels, but only 1s that are connected to 0s.
    # TODO: Do we have to redo the random sampling?
    inner_flips = np.random.choice([0, 1], size=parameters.shape, p=[1-p, p])
    inner_edges = parameters - ndimage.binary_erosion(parameters).astype(float)
    inner_update = inner_edges * inner_flips
    # Apply the inner update.
    parameters = parameters + inner_update
    parameters = np.mod(parameters, 2)
    # TODO: Check if the design is different from its parent?
    return parameters

def perturb_designs(designs_list, num_offspring, p):
    # This takes a list of original designs and outputs a list of perturbed
    # "evolved" designs.
    output_list = []
    for i in range(num_offspring):
        parameters = designs_list[i % len(designs_list)]
        # Randomly flip some pixels.
        perturbed_params = random_flips(parameters, p)
        # Binary open / close to remove islands
        perturbed_params = np.array(perturbed_params, dtype=float)
        output_list.append(perturbed_params)
    return output_list

def prune_designs(designs_list, model, num_population):
    # Kill all children except the top num_population children.
    evaluated_designs = []
    for design in designs_list:
        model.parameters = design
        objective_value = model.objective()
        evaluated_designs.append((objective_value, design))
    # sort the designs
    evaluated_designs.sort(key=lambda x:x[0])
    evaluated_designs.reverse()
    output_designs = [d for _, d in evaluated_designs[:num_population]]
    output_objectives = [v for v, _ in evaluated_designs[:num_population]]
    return output_designs, output_objectives

num_offspring = 100
num_population = 10
flip_prob = 0.1

evo_max_iter = 100

initial_design = model.parameters
designs = [initial_design]
plotting.plotGeometry(model.allParameters(), pixel_size, plot_path, 0)
for iteration in range(evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")
    new_designs = perturb_designs(designs, num_offspring, flip_prob)
    designs.extend(new_designs)  # Include the old designs in the population.
    designs, objectives = prune_designs(designs, model, num_population)

    # Every so often, delete all of the islands in the design.
    if iteration % 10 == 0:
        for i, design in enumerate(designs):
            d = ndimage.binary_opening(design)
            d = ndimage.binary_closing(d)
            designs[i] = d.astype(float)

    # Plot the best one.
    all_objective_values.append(max(objectives))

    print("Objective values:")
    for i, (v, d) in enumerate(zip(objectives, designs)):
        print(f"\t Child {i}, {v:.4f}")
    print("Best design:")
    print(designs[0])
    print("Second best design:")
    print(designs[1])
    print("\n")

    # Hack to plot the best design.
    model.parameters = designs[0]
    plotting.plotGeometry(model.allParameters(), pixel_size, plot_path, iteration+1)
    # plotGeometry(all_parameters, d, plot_path, iteration, angle1=225, angle2=45, fill_zeros=True):

print(all_objective_values)
