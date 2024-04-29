import dda_model
from dda_model import optimizers
import numpy as np
import matplotlib.pyplot as plt
import os

def construct_model():
    sym_axis = [10.5, 10.5]
    geometry_shape = [22, 22, 10]
    pixel_size = 15.0
    light_direction = [0, 0, 1]
    light_polarization = [1, 0, 0]
    wavelength = 542
    initialization = np.loadtxt("initializations/hourglass.txt")
    dielectric_constants = [1.01 + 0j, 5.96282 + 3.80423e-7j]

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

def saveCurrentStructure(model_, path, iteration):
    all_parameters = model_.allParameters()

    with open(path + '\\CoreStructure\\CoreStructure' + str(iteration) + '.txt', 'w') as f:
        for para in all_parameters:
            f.write(f"{para}\n")

def plotObjectiveFunction(objective_values, path):
    plt.figure(1)
    plt.plot(objective_values)
    #plt.legend(loc='lower right')
    plt.title('Objective Function Plot')
    plt.ylabel('Objective Function')
    plt.xlabel('Iteration #')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(path + '\\obj.png', bbox_inches='tight')

model = construct_model()
evo_max_iter = 400
# epsilon = 0.01 # This works better
epsilon = 0.01
all_objective_values = [0] * evo_max_iter

# flags for saving different values from calculation
saveObjective = True
saveStructures = True
saveEFields = True

# flags for plotting generated structures 
plotObjective = True
plotStructures = True
plotSolidStructures = False
plotZField = True
plotYField = False
plotXField = False

newpath = 'E:\\Calculations\\2024April27\\HourglassUsingWrapper2'
if not os.path.exists(newpath):
    os.makedirs(newpath + '\\CoreStructure')
    os.makedirs(newpath + '\\Shape')

optimizer = optimizers.AdamOptimizer()

for iteration in range(evo_max_iter):
    print("---------------------------------------STARTING ITERATION " + str(iteration) + "------------------------------------------")

    objective_value = model.objective()
    print("Objective Value is: " + str(objective_value))
    all_objective_values[iteration] = objective_value  

    gradients = model.gradients(objective_value)
    
    if saveStructures:
        saveCurrentStructure(model, newpath, iteration)
    gradients_final = optimizer(gradients)

    step = epsilon * gradients_final
    model.parameters = step

if plotObjective:
    plotObjectiveFunction(all_objective_values, newpath)

