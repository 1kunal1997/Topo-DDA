import plotting
import json
from pathlib import Path
import numpy as np
import itertools
import os
import sys

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

def _plotStructures(geometry, it_start, it_end, num_skip, full_path, full_lattice=False):

    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th structure')
            diel=np.genfromtxt(os.path.join(full_path, f"Structure_Values/Structure{i}.txt"))

            plotting.Shape(geometry, diel, pixel_size, iteration=i, position=full_path, FullLattice=full_lattice)

def _plotFields(geometry, it_start, it_end, num_skip, full_path):
    E_limit = True
    E_limit_low = 0
    E_limit_high = 16
    x_slice = 11
    y_slice = 11
    z_slice = 9
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th E-Field')
            E_total=np.genfromtxt(os.path.join(full_path, f"E-Field_Values/E-Field{i}.txt"),dtype=complex)

            #TODO: find a way to use the dictionary to combine these. issue is the 'Xslice' parameter
            if plot_x_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Xslice=x_slice,position=full_path)

            if plot_y_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Yslice=y_slice, position=full_path)

            if plot_z_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Zslice=z_slice, position=full_path)

            if plot_vectors:
                plotting.EField(geometry, light_direction, light_polarization, E_total, pixel_size, iteration=i, position=full_path)

def _plotPenalties(it_start, it_end, num_skip, full_path):
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th Penalty Plot')
            plotting.plotPenalties(full_path, i)

def _plotPenaltyGradients(it_start, it_end, num_skip, full_path):
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th Gradient Plot (from Penalty)')
            plotting.plotPenaltyGradients(full_path, i)

def _plotPenaltyShapes(penalty_type, full_path):
    if penalty_type == 'parabolic':
        return plotting.plotParabolic(full_path, penalty_config)
    elif penalty_type == 'gaussian':
        return plotting.plotGaussian(full_path, penalty_config)
    elif penalty_type == 'triangular':
        return plotting.plotTriangular(full_path, penalty_config)
    
def _createDirectories(path):
    plot_dict = {
            'Structure_Plots': plot_structures,
            'SolidStructure_Plots': plot_solid_structures,
            'E-Field_ZSlice_Plots': plot_z_field,
            'E-Field_YSlice_Plots': plot_y_field,
            'E-Field_XSlice_Plots': plot_x_field,
            'E-Field_Vector_Plots': plot_vectors,
            'Gradient_Penalty_Plots': plot_gradients_penalty,
            'Penalty_Plots': plot_penalties
        }

    for directory, flag in plot_dict.items():
        if flag:
            Path(os.path.join(path, directory)).mkdir(parents=True, exist_ok=True)


# CODE BEGINS HERE!!!!
full_path = sys.argv[1]
print(f"The full path is: {full_path}")
json_file = os.path.join(full_path, 'config.json')

with open(json_file) as user_file:
    parsed_json = json.load(user_file)

pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
geometry_shape = parsed_json["geometry_shape"]
evo_max_iter = parsed_json["evo_max_iteration"]

penalty_type = parsed_json["penalty_type"]
penalty_config = parsed_json["penalty_configs"][penalty_type]
coeff_type = parsed_json["coeff_type"]

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

_createDirectories(full_path)
geometry = _generateGeometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
it_start = 0
it_end = evo_max_iter
num_skip = 10

_plotPenaltyShapes(penalty_type, full_path)

if Path(os.path.join(full_path, 'Obj_Values.txt')).exists():
    plotting.plotObjectiveFunction(evo_max_iter, full_path)

if Path(os.path.join(full_path, 'Coeff_Values.txt')).exists():
    plotting.plotPenaltyCoefficients(evo_max_iter, full_path)
if Path(os.path.join(full_path, 'Penalty_Values.txt')).exists():
    plotting.plotAveragePenalty(evo_max_iter, full_path)
if Path(os.path.join(full_path, 'Structure_Values')).exists() and plot_structures:
    _plotStructures(geometry, it_start, it_end, num_skip, full_path)
if plot_solid_structures:
    _plotStructures(geometry, it_start, it_end, num_skip, full_path, full_lattice=True)
if Path(os.path.join(full_path, 'E-Field_Values')).exists() and plot_fields:
    _plotFields(geometry, it_start, it_end, num_skip, full_path)
if Path(os.path.join(full_path, 'Penalty_Values')).exists() and plot_penalties:
    _plotPenalties(it_start, it_end, num_skip, full_path)
if Path(os.path.join(full_path, 'Gradient_Penalty_Values')).exists() and plot_gradients_penalty:
    _plotPenaltyGradients(it_start, it_end, num_skip, full_path)