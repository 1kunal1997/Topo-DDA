import plotting
import json
from pathlib import Path
import numpy as np
import itertools
import os
import sys

#TODO: Duplicate method from __init__.py in dda_model. Need to make that one accessible
def _generateGeometry(num_x: int, num_y: int, num_z: int):
    # Geometry is [x y z], major along the first dimension: 0 0 0, 1 0 0, 
    # 2 0 0 ... num_x 0 0, 0 1 0, 1 1 0, ... 0 num_y 0 etc.
    mesh = itertools.product(
        list(range(num_z)),
        list(range(num_y)),
        list(range(num_x)),
    )
    geometry = np.array(list(mesh))
    # Reverse indexing required for x-major.
    geometry = geometry[:,::-1]
    geometry = geometry.flatten().astype(int)
    return geometry

def _plotStructures(geometry, it_start, it_end, num_skip, data_path, plot_path, full_lattice=False):

    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th structure')
            diel=np.genfromtxt(os.path.join(data_path, "Structures", f"Structure{i}.txt"))
            plotting.plotGeometry(geometry, diel, pixel_size, iteration=i, position=plot_path, FullLattice=full_lattice)

def _plotFields(geometry, it_start, it_end, num_skip, data_path, plot_path, config):
    E_limit = config["E_limit"]
    E_limit_low = config["E_limit_low"]
    E_limit_high = config["E_limit_high"]
    x_slice = config["x_slice"]
    y_slice = config["y_slice"]
    z_slice = config["z_slice"]
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th E-Field')
            E_total=np.genfromtxt(os.path.join(data_path, "E-Fields", f"E-Field{i}.txt"),dtype=complex)

            #TODO: find a way to use the dictionary to combine these. issue is the 'Xslice' parameter
            if plot_x_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Xslice=x_slice, position=plot_path)

            if plot_y_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Yslice=y_slice, position=plot_path)

            if plot_z_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Zslice=z_slice, position=plot_path)

            if plot_vectors:
                plotting.EField(geometry, light_direction, light_polarization, E_total, pixel_size, iteration=i, position=plot_path)

def _plotPenalties(it_start, it_end, num_skip, data_path, plot_path):
    x, penalty_shape=np.loadtxt(os.path.join(data_path, "Penalty_Shape.txt"), delimiter=',', unpack=True)
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th Penalty Plot')
            plotting.plotPenalties(x, penalty_shape, data_path, plot_path, i)

def _plotPenaltyGradients(it_start, it_end, num_skip, data_path, plot_path):
    x, penalty_gradient_shape=np.loadtxt(os.path.join(data_path, "PenaltyGradient_Shape.txt"), delimiter=',', unpack=True)
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th Gradient Plot (from Penalty)')
            plotting.plotPenaltyGradients(x, penalty_gradient_shape, data_path, plot_path, i)
    
def _createDirectories(path):
    plot_dict = {
            'Structures': plot_structures,
            'SolidStructures': plot_solid_structures,
            'E-Field_ZSlice': plot_z_field,
            'E-Field_YSlice': plot_y_field,
            'E-Field_XSlice': plot_x_field,
            'E-Field_Vector': plot_vectors,
            'Gradients_Penalty': plot_gradients_penalty,
            'Penalties': plot_penalties
        }

    for directory, flag in plot_dict.items():
        if flag:
            Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)

# CODE BEGINS HERE!!!!
full_path = sys.argv[1]
print(f"The full path is: {full_path}")
json_file = os.path.join(full_path, 'config.json')

data_path = os.path.join(full_path, "Data")
plot_path = os.path.join(full_path, "Plots")

with open(json_file) as user_file:
    parsed_json = json.load(user_file)

pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
geometry_shape = parsed_json["geometry_shape"]
evo_max_iter = parsed_json["evo_max_iteration"]

penalty_type = parsed_json["penalty_type"]
penalty_config = parsed_json["penalty_configs"][penalty_type]

# plotting flags
plot_structures = parsed_json["plot_structures"]
plot_solid_structures = parsed_json["plot_solid_structures"]
plot_fields = parsed_json["plot_fields"]             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plot_z_field = parsed_json["plot_z_field"]
plot_y_field = parsed_json["plot_y_field"]
plot_x_field = parsed_json["plot_x_field"]
plot_vectors = parsed_json["plot_vectors"]

#other plotting stuff
plot_gradients_penalty = parsed_json["plot_gradients_penalty"]
plot_penalties = parsed_json["plot_penalties"]

_createDirectories(plot_path)
geometry = _generateGeometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
it_start = parsed_json["it_start"]
it_end = parsed_json["it_end"]
num_skip = parsed_json["num_skip"]

#_plotPenaltyShapes(penalty_type, data_path, plot_path)
plotting.plotPenaltyShape(penalty_type, data_path, plot_path, penalty_config)
plotting.plotStepSizes(evo_max_iter, data_path, plot_path)
plotting.plotObjectiveFunction(evo_max_iter, data_path, full_path)
plotting.plotPenaltyCoefficients(evo_max_iter, data_path, plot_path)
plotting.plotAveragePenalty(evo_max_iter, data_path, plot_path)

if plot_structures:
    _plotStructures(geometry, it_start, it_end, num_skip, data_path, plot_path)
if plot_solid_structures:
    _plotStructures(geometry, it_start, it_end, num_skip, data_path, plot_path, full_lattice=True)
if plot_fields:
    _plotFields(geometry, it_start, it_end, num_skip, data_path, plot_path, parsed_json["EField_config"])
if plot_penalties:
    _plotPenalties(it_start, it_end, num_skip, data_path, plot_path)
if plot_gradients_penalty:
    _plotPenaltyGradients(it_start, it_end, num_skip, data_path, plot_path)