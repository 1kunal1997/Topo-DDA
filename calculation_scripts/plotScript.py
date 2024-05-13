import plotting
import json
from pathlib import Path
import numpy as np
import itertools
import os

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

def _plotStructures(geometry, it_start, it_end, num_skip, new_path, full_lattice=False):

    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th structure')
            diel=np.genfromtxt(os.path.join(new_path, f"Structure_Values/Structure{i}.txt"))

            plotting.Shape(geometry, diel, pixel_size, iteration=i, position=new_path, FullLattice=full_lattice)

def _plotFields(geometry, it_start, it_end, num_skip, new_path):
    E_limit = True
    E_limit_low = 0
    E_limit_high = 16
    x_slice = 11
    y_slice = 11
    z_slice = 9
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th E-Field')
            E_total=np.genfromtxt(os.path.join(new_path, f"E-Field_Values/E-Field{i}.txt"),dtype=complex)

            #TODO: find a way to use the dictionary to combine these. issue is the 'Xslice' parameter
            if plot_x_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Xslice=x_slice,position=new_path)

            if plot_y_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Yslice=y_slice, position=new_path)

            if plot_z_field:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=E_limit, Elimitlow=E_limit_low, Elimithigh=E_limit_high, iteration=i, Zslice=z_slice, position=new_path)

            if plot_vectors:
                plotting.EField(geometry, light_direction, light_polarization, E_total, pixel_size, iteration=i, position=new_path)

with open('config.json') as user_file:
    parsed_json = json.load(user_file)

pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
geometry_shape = parsed_json["geometry_shape"]
evo_max_iter = parsed_json["evo_max_iteration"]
step_size = parsed_json["step_size"]
coeff_min = parsed_json["coeff_min"]
coeff_max = parsed_json["coeff_max"]
base_path = parsed_json["base_path"]

penalty_list = parsed_json["penalty_list"]
coeff_list = parsed_json["coeff_list"]

plot_objective = parsed_json["plot_objective"]
plot_structures = parsed_json["plot_structures"]
plot_solid_structures = parsed_json["plot_solid_structures"]
plot_fields = parsed_json["plot_fields"]             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plot_z_field = parsed_json["plot_z_field"]
plot_y_field = parsed_json["plot_y_field"]
plot_x_field = parsed_json["plot_x_field"]
plot_vectors = parsed_json["plot_vectors"]

for penalty_type in penalty_list:
    for coeff_type in coeff_list:

        new_path = base_path + '_it' + str(evo_max_iter) + '_eps' + str(step_size) + '_penalty_' + penalty_type + '_coeff' + str(coeff_min) + 'to' + str(coeff_max) + '_' + coeff_type
        plot_dict = {
                    '\\Structures': plot_structures,
                    '\\SolidStructures': plot_solid_structures,
                    '\\E-Field_ZSlice': plot_z_field,
                    '\\E-Field_YSlice': plot_y_field,
                    '\\E-Field_XSlice': plot_x_field,
                    '\\E-Field_Vectors': plot_vectors   
                }

        for directory, flag in plot_dict.items():
            if flag:
                Path(new_path + directory).mkdir(parents=True, exist_ok=True)

        geometry = _generateGeometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        it_start = 0
        it_end = evo_max_iter
        num_skip = 5

        if plot_objective:
            plotting.plotObjectiveFunction(evo_max_iter, new_path)
        if plot_structures:
            _plotStructures(geometry, it_start, it_end, num_skip, new_path)
        if plot_solid_structures:
            _plotStructures(geometry, it_start, it_end, num_skip, new_path, full_lattice=True)
        if plot_fields:
            _plotFields(geometry, it_start, it_end, num_skip, new_path)