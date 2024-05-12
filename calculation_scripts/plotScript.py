import plotting
import json
from pathlib import Path
import numpy as np
import itertools
import os

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

with open('config.json') as user_file:
    parsed_json = json.load(user_file)

pixel_size = parsed_json["pixel_size"]
light_direction = parsed_json["light_direction"]
light_polarization = parsed_json["light_polarization"]
geometry_shape = parsed_json["geometry_shape"]
evo_max_iter = parsed_json["evo_max_iteration"]
epsilon = parsed_json["epsilon"]
coeff_min = parsed_json["coeff_min"]
coeff_max = parsed_json["coeff_max"]
base_path = parsed_json["base_path"]

penaltyList = parsed_json["penaltyList"]
coeffList = parsed_json["coeffList"]

plotObjective = parsed_json["plotObjective"]
plotStructures = parsed_json["plotStructures"]
plotSolidStructures = parsed_json["plotSolidStructures"]
plotEFields = parsed_json["plotEFields"]             # umbrella flag for no E-Field plots at all

# Specific E-field plot flags
plotZField = parsed_json["plotZField"]
plotYField = parsed_json["plotYField"]
plotXField = parsed_json["plotXField"]
plotEVectors = parsed_json["plotEVectors"]

for penaltyType in penaltyList:
    for coeffType in coeffList:

        newpath = base_path + '_it' + str(evo_max_iter) + '_eps' + str(epsilon) + '_penalty_' + penaltyType + '_coeff' + str(coeff_min) + 'to' + str(coeff_max) + '_' + coeffType
        plotDict = {
                    '\\Structures': plotStructures,
                    '\\SolidStructures': plotSolidStructures,
                    '\\E-Field_ZSlice': plotZField,
                    '\\E-Field_YSlice': plotYField,
                    '\\E-Field_XSlice': plotXField,
                    '\\E-Field_Vectors': plotEVectors   
                }

        for directory, flag in plotDict.items():
            if flag:
                Path(newpath + directory).mkdir(parents=True, exist_ok=True)

        geometry = _generate_geometry(geometry_shape[0], geometry_shape[1], geometry_shape[2])
        itStart = 0
        itEnd = evo_max_iter
        numskip = 5

        if plotObjective:
            plotting.plotObjectiveFunction(evo_max_iter, newpath)
        if plotStructures:
            plotStructuresFunction(geometry, itStart, itEnd, numskip)
        if plotSolidStructures:
            plotStructuresFunction(geometry, itStart, itEnd, numskip, solid=True)
        if plotEFields:
            plotFieldsFunction(geometry, itStart, itEnd, numskip)