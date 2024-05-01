import dda_model
from dda_model import optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import plotting
import itertools

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

def _generate_geometry(num_x: int, num_y: int, num_z: int):
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

def plotStructuresFunction(geometry, itStart, itEnd, numskip, solid=False):
    print('new path is: ' + newpath)
    for i in range(itStart, itEnd):
        if i % numskip == 0:
            print('Plotting the ' + str(i) + 'th structure')
            CoreStructure=np.genfromtxt(os.path.join(newpath+"\\Structure_Values\\Structure"+str(i)+".txt"),dtype=complex)
            diel=np.real(CoreStructure)

            if solid:
                plotting.Shape(geometry, diel, pixel_size, iteration=i, position=newpath+"\\SolidStructures\\", FullLattice=True)
            else:
                plotting.Shape(geometry, diel, pixel_size, iteration=i, position=newpath+"\\Structures\\", FullLattice=False)

def plotFieldsFunction(geometry, itStart, itEnd, numskip):
    ELimit = True
    ELimitLow = 0
    ELimitHigh = 16
    xslice = 11
    yslice = 11
    zslice = 9
    for i in range(itStart, itEnd):
        if i % numskip == 0:
            print('Plotting the ' + str(i) + 'th E-Field')
            E_total=np.genfromtxt(os.path.join(newpath+"\\E-Field_Values\\E-Field"+str(i)+".txt"),dtype=complex)

            #TODO: find a way to use the dictionary to combine these. issue is the 'Xslice' parameter
            if plotXField:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Xslice=xslice,position=newpath+"\\E-Field_XSlice\\")

            if plotYField:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Yslice=yslice, position=newpath+"\\E-Field_YSlice\\")

            if plotZField:
                plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Zslice=zslice, position=newpath+"\\E-Field_ZSlice\\")

            if plotEVectors:
                plotting.EField(geometry,light_direction, light_polarization, E_total, pixel_size, iteration=i, position=newpath+"\\E-Field_Vectors\\")

sym_axis = [10.5, 10.5]
geometry_shape = [22, 22, 10]
pixel_size = 15.0
light_direction = [0, 0, 1]
light_polarization = [1, 0, 0]
wavelength = 542
initialization = np.loadtxt("initializations/hourglass.txt")
#initialization += np.random.uniform(0, 10e-3, size=initialization.shape)
dielectric_constants = [1.01 + 0j, 5.96282 + 3.80423e-7j]

# flags for saving different values from calculation
saveObjective = True
saveStructures = True
saveEFields = True

plotFlag = True                # umbrella flag for no plots except objfunc
# flags for plotting generated structures 
plotObjective = True
plotStructures = True
plotSolidStructures = True
plotEFields = True             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plotZField = True
plotYField = True
plotXField = True
plotEVectors = True

for i in range(1):
    model = construct_model()
    evo_max_iter = 400
    # epsilon = 0.01 # This works better
    epsilon = 0.01
    all_objective_values = [0] * evo_max_iter

    newpath = 'E:\\Calculations\\2024April30\\HourglassWithPlottingDictionary2'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    plotDict = {
        '\\Structure_Values': saveStructures,
        '\\Structures': plotStructures,
        '\\SolidStructures': plotSolidStructures,
        '\\E-Field_Values': saveEFields,
        '\\E-Field_ZSlice': plotZField,
        '\\E-Field_YSlice': plotYField,
        '\\E-Field_XSlice': plotXField,
        '\\E-Field_Vectors': plotEVectors
         
    }

    for directory, flag in plotDict.items():
        if flag and not os.path.exists(newpath + directory):
            os.makedirs(newpath + directory)

    optimizer = optimizers.AdamOptimizer()
    # main iteration loop for gradient descent optimization
    for iteration in range(evo_max_iter):
        print("---------------------------------------STARTING ITERATION " + str(iteration) + "------------------------------------------")

        objective_value = model.objective()
        print("Objective Value is: " + str(objective_value))
        all_objective_values[iteration] = objective_value  

        gradients = model.gradients(objective_value)
        
        if saveStructures:
            all_parameters = model.allParameters()
            saveCurrentStructure(all_parameters, newpath, iteration)
        if saveEFields:
            electricField = model.getElectricField()
            saveCurrentEField(electricField, newpath, iteration)

        gradients_final = optimizer(gradients)

        step = epsilon * gradients_final
        model.parameters = step
    
    # plotting. may need to be moved in a function, or into plotting.py 
    if saveObjective:
        saveObjectiveFunction(all_objective_values, newpath)
    if plotObjective:
        plotting.plotObjectiveFunction(all_objective_values, newpath)
    
    if plotFlag:
        geometry = _generate_geometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        itStart = 0
        itEnd = 400
        numskip = 5

        if saveStructures and plotStructures:
            plotStructuresFunction(geometry, itStart, itEnd, numskip)
        if saveStructures and plotSolidStructures:
            plotStructuresFunction(geometry, itStart, itEnd, numskip, solid=True)
        if saveEFields and plotEFields:
            plotFieldsFunction(geometry, itStart, itEnd, numskip)
        
    '''
    if saveStructures and plotStructures:

        geometry = _generate_geometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        itStart = 0
        itEnd = 400
        numskip = 5
        for i in range(itStart, itEnd):
            if i % numskip == 0:
                print('Plotting the ' + str(i) + 'th structure')
                CoreStructure=np.genfromtxt(os.path.join(newpath+"\\Structure_Values\\Structure"+str(i)+".txt"),dtype=complex)
                diel=np.real(CoreStructure)

                plotting.Shape(geometry, diel, pixel_size, iteration=i, position=newpath+"\\Structures\\", FullLattice=False)

    if saveEFields and plotEFields:
        geometry = _generate_geometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        itStart = 0
        itEnd = 400
        numskip = 5
        ELimit = True
        ELimitLow = 0
        ELimitHigh = 16
        xslice = 11
        yslice = 11
        zslice = 9
        for i in range(itStart, itEnd):
            if i % numskip == 0:
                print('Plotting the ' + str(i) + 'th E-Field')
                E_total=np.genfromtxt(os.path.join(newpath+"\\E-Field_Values\\E-Field"+str(i)+".txt"),dtype=complex)

                if plotXField:
                    plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Xslice=xslice,position=newpath+"\\E-Field_Slices\\")

                if plotYField:
                    plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Yslice=yslice, position=newpath+"\\E-Field_Slices\\")

                if plotZField:
                    plotting.EField_slice(geometry, E_total, pixel_size, Elimit=ELimit, Elimitlow=ELimitLow, Elimithigh=ELimitHigh, iteration=i, Zslice=zslice, position=newpath+"\\E-Field_Slices\\")

    if saveEFields and plotEVectors:
        geometry = _generate_geometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        itStart = 0
        itEnd = 400
        numskip = 5
        for i in range(itStart, itEnd):
            if i % numskip == 0:
                print('Plotting the ' + str(i) + 'th E-Field Vectors')
                E_total=np.genfromtxt(os.path.join(newpath+"\\E-Field_Values\\E-Field"+str(i)+".txt"),dtype=complex)

                plotting.EField(geometry,light_direction, light_polarization, E_total, pixel_size, iteration=i, position=newpath+"\\E-Field_Vectors\\")
    '''



