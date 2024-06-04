import plotting
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def _plotStructures(it_start, it_end, num_skip, data_path, plot_path, fill_zeros=False):
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th structure')
            diel=np.load(os.path.join(data_path, "Structures", f"Structure{i}.npy"))
            if fill_zeros:
                path = os.path.join(plot_path, "Structures")
            else:
                path = os.path.join(plot_path, "SolidStructures")
            plotting.plotGeometry(diel, pixel_size, path, i, fill_zeros=fill_zeros)
            plt.close()


def _plotFields(it_start, it_end, num_skip, data_path, plot_path, config):
    E_limit = config["E_limit"]
    E_limit_low = config["E_limit_low"]
    E_limit_high = config["E_limit_high"]
    display_limits = [E_limit_low, E_limit_high] if E_limit else None
    x_slice = config["x_slice"]
    y_slice = config["y_slice"]
    z_slice = config["z_slice"]
    for i in range(it_start, it_end):
        if i % num_skip == 0:
            print(f'Plotting the {i}th E-Field')
            E_total=np.load(os.path.join(data_path, "E-Fields", f"E-Field{i}.npy"))
            print(E_total.shape)

            #TODO: find a way to use the dictionary to combine these. issue is the 'Xslice' parameter
            if plot_x_field:
                plotting.EField_slice(E_total, os.path.join(plot_path, "E-Field_XSlice"), i, index=x_slice, axis='x', cbar_limits=display_limits)
                plt.close()
            if plot_y_field:
                plotting.EField_slice(E_total, os.path.join(plot_path, "E-Field_YSlice"), i, index=y_slice, axis='y', cbar_limits=display_limits)
                plt.close()
            if plot_z_field:
                plotting.EField_slice(E_total, os.path.join(plot_path, "E-Field_ZSlice"), i, index=z_slice, axis='z', cbar_limits=display_limits)
                plt.close()
    
def _createDirectories(path):
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

# plotting flags
plot_structures = parsed_json["plot_structures"]
plot_solid_structures = parsed_json["plot_solid_structures"]
plot_fields = parsed_json["plot_fields"]             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plot_z_field = parsed_json["plot_z_field"]
plot_y_field = parsed_json["plot_y_field"]
plot_x_field = parsed_json["plot_x_field"]

_createDirectories(plot_path)
it_start = parsed_json["it_start"]
it_end = min(parsed_json["it_end"], evo_max_iter)
num_skip = parsed_json["num_skip"]

plotting.plotObjectiveFunction(evo_max_iter, data_path, full_path)
plotting.plotStepSizes(evo_max_iter, data_path, plot_path)

if plot_structures:
    _plotStructures(it_start, it_end, num_skip, data_path, plot_path, fill_zeros=True)
if plot_solid_structures:
    _plotStructures(it_start, it_end, num_skip, data_path, plot_path, fill_zeros=False)
if plot_fields:
    _plotFields(it_start, it_end, num_skip, data_path, plot_path, parsed_json["EField_config"])
