import numpy as np
from scipy import ndimage
import dda_model
import matplotlib.pyplot as plt

domain_shape = [22, 22, 10]
parameter_shape = [11, 11]

num_inits = 10
num_success = 0
while num_success < num_inits:
    X = np.random.choice([0, 1], size=parameter_shape, p=[0.9, 0.1])
    if X.sum() <= 0:
        continue
    num_success += 1
    # Apply a dilation to the initialization.
    X = ndimage.binary_dilation(X, iterations=2)
    X = X.astype(int)
    X = np.concatenate([X, X[::-1,:]], axis = 0)
    X = np.concatenate([X[:,::-1], X], axis = 1)
    # Reformat into the stupid format, to write the output file.
    geometry = dda_model._generate_geometry(*domain_shape).reshape([-1,3])
    with open(f'initializations/random{num_success}.txt', 'wt') as f:
        for xyz in geometry:
            for _ in range(3):
                x,y,z = xyz
                f.write(str(X[x,y]))
                f.write('\n')

