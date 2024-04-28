import dda_model
import numpy as np
from scipy.special import sici
import pytest
import time


def _construct_test_model():
    # Filter parameters.
    # Note: These should be disabled as filters are being moved to Python.
    filter_beta_min = 0.0
    filter_beta_max = 50.0
    filter_ita = 0.5
    filter_method = "piecewise"
    filter_iterations = [100]
    filter_radii = [2.0]
    filter_enable = False

    # Symmetry parameters.
    sym_method = "4fold"
    sym_axis = [10.5, 10.5]
    sym_is_periodic = True

    # Objective function parameters.
    obj_name = "IntegratedE"
    obj_config = [2.0, 0.0, 21.0, 0.0, 21.0, 0.0, 9.0, 0.95, 50.0]

    # Geometry parameters.
    geometry_config = np.loadtxt("data/geometry.txt", dtype=int)
    geo_indices = geometry_config[4:]
    geo_nx, geo_ny, geo_nz, geo_ntotal = geometry_config[:4]
    geo_pixel_size_nm = 15.0

    # Dielectric (optimizable value) parameters.
    dielectric_values = np.loadtxt("data/dielectrics.txt")
    dielectric_materials = [1.01 + 0j, 5.96282 + 3.80423e-7j]

    # Incident light wave parameters.
    light_direction = [0, 0, 1]
    light_polarization = [1, 0, 0]
    light_amplitude = 1.0
    light_wavelength_nm = 542
    background_refractive_index = np.sqrt(
        np.real(dielectric_materials[0]))

    # Numerical optimization parameters.
    # A-product-core periodicity parameters / boundaries.
    apc_max_m = 50
    apc_max_n = 50
    apc_l_m = 22
    apc_l_n = 22
    apc_method_name = "FCD"

    # Cached integral values.
    sici_delta = 0.1
    sici_n = 1_000_000
    integral_positions = np.linspace(sici_delta, sici_delta * sici_n, sici_n - 1)
    si, ci = sici(integral_positions)

    ordered_parameters = {
        # Filter parameters.
        "filter_beta_min": filter_beta_min,
        "filter_beta_max": filter_beta_max,
        "filter_ita": filter_ita,
        "filter_method": filter_method,
        "filter_iterations": filter_iterations,
        "filter_radii": filter_radii,
        "filter_enable": filter_enable,
        # Symmetry parameters.
        "sym_method": sym_method,
        "sym_axis": sym_axis,
        "sym_is_periodic": sym_is_periodic,
        # Objective and geometry.
        "obj_name": obj_name,
        "obj_config": obj_config,
        "geo_indices": geo_indices,
        "dielectric_values": dielectric_values,
        "geo_nx": geo_nx,
        "geo_ny": geo_ny,
        "geo_nz": geo_nz,
        "geo_ntotal": geo_ntotal,
        # Incident light wave (plus some material properties).
        "light_direction": light_direction,
        "light_amplitude": light_amplitude,
        "light_polarization": light_polarization,
        "light_wavelength_nm": light_wavelength_nm,
        "dielectric_materials": dielectric_materials,
        "background_refractive_index": background_refractive_index,
        # Numerical parameters.
        "apc_max_m": apc_max_m,
        "apc_max_n": apc_max_n,
        "apc_l_m": apc_l_m,
        "apc_l_n": apc_l_n,
        "apc_method_name": apc_method_name,
        "geo_pixel_size_nm": geo_pixel_size_nm,
        # Sine and cosine integral values.
        "sineIntegralValues": si,
        "cosineIntegralValues": ci,
        "integralDelta": sici_delta,
        "verbose": False,
    }

    model = dda_model.DDAModel(*list(ordered_parameters.values()))
    return model, ordered_parameters


GLOBAL_MODEL = _construct_test_model()

def test_objective_calculation():
    # This tests at the lower-level C++ API (not the wrapped API)
    model, parameters = GLOBAL_MODEL
    origin_polarization = [0 + 0j] * 3 * parameters["geo_ntotal"]
    # Numerical parameters for the bicongugate gradient stabilized method. 
    bgs_max_iter = 100_000
    bgs_max_error = 1e-5
    # Must call .solveElectricField before calculating the objective.
    t_start = time.time()
    model.solveElectricField(origin_polarization, 100_000, 1e-5)
    print(f"Took {time.time() - t_start:.3f} seconds to solve the E field.")
    t_start = time.time()
    objective_value = model.calculateObjective()
    print("Objective Value is: " + str(objective_value))

def test_gradients_calculation():
    # model, parameters = _construct_test_model()
    model, parameters = GLOBAL_MODEL
    origin_polarization = [0 + 0j] * 3 * parameters["geo_ntotal"]
    # Numerical parameters for the bicongugate gradient stabilized method. 
    bgs_max_iter = 100_000
    bgs_max_error = 1e-5
    # Must call .solveElectricField before calculating the objective.
    model.solveElectricField(origin_polarization, 100_000, 1e-5)
    objective_value = model.calculateObjective()    
    epsilon_partial = 0.001
    gradients = model.calculateGradients(epsilon_partial, objective_value, bgs_max_iter, bgs_max_error)
    print(gradients)
