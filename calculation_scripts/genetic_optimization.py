import dda_model

import numpy as np
import json
from pathlib import Path
import os
import sys
from scipy import ndimage

import matplotlib.pyplot as plt
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
    directories = ['Structures', 'Parameters', 'E-Fields']
    for directory in directories:
        Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

def _createPlotDirectories(path):
    plot_dict = {
            'Structures': plot_structures,
            'SolidStructures': plot_solid_structures,
            'E-Field_ZSlice': plot_z_field,
            'E-Field_YSlice': plot_y_field,
            'E-Field_XSlice': plot_x_field,
        }

    for directory, flag in plot_dict.items():
        if flag:
            Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

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

# plotting flags
plot_structures = parsed_json["plot_structures"]
plot_solid_structures = parsed_json["plot_solid_structures"]
plot_fields = parsed_json["plot_fields"]             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plot_z_field = parsed_json["plot_z_field"]
plot_y_field = parsed_json["plot_y_field"]
plot_x_field = parsed_json["plot_x_field"]

base_path = parsed_json["base_path"]
run_name = parsed_json["run_name"]
full_path = os.path.join(base_path, run_name)
print("Saving value to path: " + full_path)
data_path = os.path.join(full_path, "Data")
plot_path = os.path.join(full_path, "Plots")
Path(plot_path).mkdir(parents=True, exist_ok=True)
_createDirectories(data_path)
_createPlotDirectories(plot_path)


model = _constructModel()
all_objective_values = []

# Calculate objective value of initialization and save data as iteration 0
objective_value = model.objective()
parameters = model.parameters
print(f"Objective value of initialization is: {objective_value}")
all_objective_values.append(objective_value)
_saveCurrentStructure(model.allParameters(), data_path, 0)
_saveCurrentEField(model.getElectricField(), data_path, 0)
_saveAllParams(model.parameters, data_path, 0)

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

def dedupe_designs(designs_list):
    # Remove identical designs. This may or may not help optimization.
    unique_designs = []
    for design in designs_list:
        found_dupe = False
        for unique_design in unique_designs:
            if np.sum((unique_design - design)**2) < 10e-6:
                found_dupe = True
                break
        if not found_dupe:
            unique_designs.append(design)
    return unique_designs

def prune_designs(designs_list, model, num_population, remove_islands=False, max_island_size=1):
    # Kill all children except the top num_population children.
    def padded_morph_op(x, operation):
        # operation: one of ndimage.binary_opening, binary_closing, etc.
        n = max_island_size
        padded = np.pad(x, n, mode='edge')
        output = operation(padded, iterations=n)
        return output.astype(float)[n:-n,n:-n]
    opening = lambda x: padded_morph_op(x, ndimage.binary_opening)
    closing = lambda x: padded_morph_op(x, ndimage.binary_closing)

    evaluated_designs = []
    # This can actually hurt quality but is good for exploration.
    # designs_list = dedupe_designs(designs_list)
    for design in designs_list:
        if remove_islands:
            design = closing(opening(design)).astype(float)
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
flip_prob = 0.05
filter_cadence = 20  # Design constraints are enforced every "cadence" iters.
evo_max_iter = parsed_json["evo_max_iteration"]


initial_design = model.parameters
designs = [initial_design]

'''
# Generate random initializations as the initial population.
parameter_shape = [11, 11]

designs = []
for _ in range(num_population):
    init_successful = False
    while not init_successful:
        X = np.random.choice([0, 1], size=parameter_shape, p=[0.95, 0.05])
        # Apply a (random) dilation to the initialization.
        n_iters = np.random.choice([0, 1, 2])
        X = ndimage.binary_dilation(X, iterations=n_iters).astype(float)
        if X.sum() <= 10e-6:
            continue
        if np.sum((X - np.ones_like(X))**2) <= 10e-6:
            continue
        init_successful = True
        designs.append(X)
'''

last_design_plotted = 0
num_plotted = 0

# Perform the optimization loop.
for iteration in range(evo_max_iter):
    print(f"{'-' * 40}STARTING ITERATION {iteration} {'-' * 40}")
    new_designs = perturb_designs(designs, num_offspring, flip_prob)
    designs.extend(new_designs)  # Include the old designs in the population.
    remove_islands = (iteration%filter_cadence == 0)
    designs, objectives = prune_designs(
        designs, model, num_population,
        remove_islands=remove_islands,
        max_island_size=1,
    )

    # Plot the best one.
    all_objective_values.append(max(objectives))

    print("Objective values:")
    for i, (v, d) in enumerate(zip(objectives, designs)):
        print(f"\t Child {i}, {v:.4f}")
    print("")

    # Hack to plot the best design.
    should_skip_plot = (np.sum((last_design_plotted - designs[0])**2) < 10e-6)
    if not should_skip_plot:
        # This plot differs from the previous one.
        model.parameters = designs[0]
        plotting.plotGeometry(model.allParameters(), pixel_size, os.path.join(plot_path, "Structures"), num_plotted)
        # Hacky way to re-save the plot but with a new title.
        plt.suptitle(f'Iteration {iteration+1}, Obj={max(objectives):.4f}')
        plt.savefig(os.path.join(plot_path, "Structures", f"Structure{num_plotted}.png"), dpi=100)
        plt.close()
        #np.save(os.path.join(plot_path, f"design_{num_plotted}.npy"), designs[0])
        _saveCurrentStructure(model.allParameters(), data_path, iteration)
        _saveCurrentEField(model.getElectricField(), data_path, iteration)
        _saveAllParams(model.parameters, data_path, iteration)
        num_plotted += 1
        last_design_plotted = designs[0]

_saveObjective(all_objective_values, data_path)
plotting.plotObjectiveFunction(evo_max_iter, data_path, plot_path)
