import numpy as np
from scipy import ndimage
import dda_model
import matplotlib.pyplot as plt
import sys
import json
import os


# Call as: python script.py base_config.json output.json integer_run_id

base_config_filename = sys.argv[1]
out_config_filename = sys.argv[2]
run_id = sys.argv[3]

# Load the default config options.
with open(base_config_filename) as base_config_file:
    print("config: ",base_config_filename)
    base_config = json.load(base_config_file)

# Update base config with some extra options.
base_path = base_config["base_path"]
run_name = f"run_{run_id}"
full_path = os.path.join(base_path, run_name)
init_path = os.path.join(full_path, "initialization.txt")

base_config["full_path"] = full_path
base_config["run_name"] = run_name
base_config["init_path"] = init_path

# Save the updated config to the desired location.
with open(out_config_filename, 'w') as out_config_file:
    json.dump(base_config, out_config_file)

# Generate the initialization and save it to the init_path location.
domain_shape = [22, 22, 10]
parameter_shape = [11, 11]

init_successful = False
while not init_successful:
    X = np.random.choice([0, 1], size=parameter_shape, p=[0.95, 0.05])
    if X.sum() <= 0:
        continue
    init_successful = True
    # Apply a dilation to the initialization.
    n_iters = np.random.choice([0, 1, 2])
    X = ndimage.binary_dilation(X, iterations=n_iters)
    X = X.astype(int)
    X = np.concatenate([X, X[::-1,:]], axis = 0)
    X = np.concatenate([X[:,::-1], X], axis = 1)
    # Reformat into the stupid format, to write the output file.
    geometry = dda_model._generate_geometry(*domain_shape).reshape([-1,3])
    # Make the calculation directory if it does not exist.
    os.makedirs(os.path.dirname(init_path), exist_ok=True)
    with open(init_path, 'wt') as f:
        for xyz in geometry:
            for _ in range(3):
                x,y,z = xyz
                f.write(str(X[x,y]))
                f.write('\n')
