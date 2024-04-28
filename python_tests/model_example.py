import dda_model
import numpy as np
from scipy.special import sici

'''

Task.ini: 

name=EvoOpt 2D input perioidic

From DDA_verify:
Nx=31
Ny=31
Nz=31
r=30

material:
material1=Air
material2=SiO2

Grid:
d=2
lam=500

MAX_ITERATION_DDA=10000
MAX_ERROR=0.00001




'''



'''
betamin_, betamax_, ita_, betatype: Used to update beta, which is used in the SmoothDensity function

filterIterations_: vector of ints (size determined by user) for which iteration to change the filter radius. 
    used to construct 'filterinfo' struct.

filterRadii_: one-to-one mapping to 'filterIterations_' for setting the radii.

Filter_: boolean that determines whether you are using the internal Filter or not.

symmetry: string to turn on or off "4-fold" symmetry. to add other options for symmetry, look up 'dividesym'.

symaxis: vector of length 2 that sets the x and y coordinates for the axes of symmetry.

Periodic_: boolean for whether you want a periodic structure (in xy) or not.

objName_: name of the objective function to use.

objPara_: set of parameters needed for the corresponding obj function. size depends on obj function.

geometry_: vector of ints of length 3*N. numbering system for each pixel. (0, 0, 0, 1, 0, 0, 2, 0, 0, ...)

InputDiel_: vector of doubles of length 3*N. initialization of the pixel values (parameters)

Nx_, Ny_, Nz_,: ints for the number of pixels in the x, y, and z direction. 

N_: total number of pixels in structure

n_K_: vector of doubles of length 3. direction that incoming plane wave of light is propagating in.

n_E0_: vector of doubles of length 3. Defines the polarization of light.

E0_: amplitude of the incoming plane wave.

lam_: wavelength of the incoming plane wave.

material_: vector of complex doubles of length 2. first element is the dielectric function of external domain,
    second element is the dielectric function of the material of interest. determined using interpolation.

nback_: refractive index of background.

MAXm_, MAXn_: integers used in AProductCore for periodicity reasons.

Lm_, Ln_: integers for periodic boundary in the x and y direction, respectively.

AMatrixMethod_: string used to determine which method to use in AProductCore.

d_: size of pixel in nanometers.
'''

beta_min = 0
beta_max = 1
ita = 0
betatype = "piecewise"
useFilter = False

symmetryType = "4fold"
symmetryAxis = [10.5, 10.5]
isPeriodic = True


# objPara is a shitty configuration containing multiple pieces of shit.
# First number is integration power (E^number)
# Next 6 numbers are integration bounds in each dimension (x_min, x_max, y_min, y_max, z_min, z_max)
# Next two numbers have to do with the smoothing function. The filter function (that we are not using presently)
# They are quite optional, depend on betatype etc.

integration_config = {
    "power": 2,
    "x_bounds": [0, 21],
    "y_bounds": [0, 21],
    "z_bounds": [0, 9],
    "unused_field": [-1, -1],
}

# geometry, inputDielectrics
# both 1-dimensional and of size 3N

dims_and_geometry = np.loadtxt("data/geometry.txt", dtype=int)
geometry = dims_and_geometry[4:]
nx, ny, nz, ntotal = dims_and_geometry[:4]
print(nx, ny, nz, ntotal)
print(geometry)
inputDielectrics = np.loadtxt("data/dielectrics.txt")

# n_k is the direction of propagation for the input light
n_k = [0, 0, 1]
# n_E0 is the polarization of the input light
n_E0 = [1, 0, 0]
# E0 is the magnitude
E0 = 1.0
# lam is the wavelength (in nm)
lam = 542
# material:
# 1 dimensional vector with 2 values
# first value is the background dielectric constant
# and the 2nd is the dielectric constant of the material of interest
# We have to interpolate the (wavelength, value) pulled from the literature / internet / whatever
# normally we would do an interpolation and load this from a file, but for this test
# we are going to just use some hardcoded (slightly wrong) values. To test that the
# bindings work etc
material = [1.01 + 0j, 5.96282 + 3.80423e-7j]

# Background material - some constant that idk the physical interpretation of 
nback = np.sqrt(np.real(material[0]))

# Numerical convergence parameters for 
MAXm = 50
MAXn = 50
# Lm and Ln are the length and width of the structure - this can probably be computed
# based on other inputs.
Lm = 22
Ln = 22
# matrixAMethodName: eligible choices "FCD" and "LDR" 
matrixAMethodName = "FCD"

# d: pixel size in nm
d = 15.0

# Sine and cosine integral values.
sici_delta = 0.1
sici_n = 1_000_000
integral_positions = np.linspace(sici_delta, sici_delta * sici_n, sici_n - 1)
si, ci = sici(integral_positions)

model = py_dda_model.DDAModel(
    0.0, 50.0, 0.5, "piecewise", [100], [2.0], False,
    "4fold", [10.5, 10.5], True,
    "IntegratedE", [2.0, 0.0, 21.0, 0.0, 21.0, 0.0, 9.0, 0.95, 50.0],
    geometry, inputDielectrics, nx, ny, nz, ntotal,
    n_k, E0, n_E0, lam, material, 
    nback, MAXm, MAXn, Lm, Ln, matrixAMethodName, d, 
    si, ci, sici_delta,
    True,
)

print(model)

print('*'*20)
print("Solving electric field")
print('*'*20)

originPolarization = [0 + 0j] * 3 * ntotal
originPolarization = np.array(originPolarization)
model.solveElectricField(originPolarization, 100_000, 1e-5)

print('*'*20)
print("Calculating Objective")
print('*'*20)

objective = model.calculateObjective()

print("Objective value = ", objective)




