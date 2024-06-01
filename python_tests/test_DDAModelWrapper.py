import dda_model
import numpy as np
from scipy.special import sici
import pytest
import time


def _construct_test_model():
    sym_axis = [10.5, 10.5]
    geometry_shape = [22, 22, 10]
    pixel_size = 15.0
    light_direction = [0, 0, 1]
    light_polarization = [1, 0, 0]
    wavelength = 542
    initialization = np.loadtxt("data/dielectrics.txt")
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
        verbose=False,
    )
    return model


def test_wrapper_objective():
    t_start = time.time()
    model = _construct_test_model()
    print(f"Took {time.time() - t_start:.3f} seconds to initialize the model.")
    t_start = time.time()
    objective_value = model.objective()
    print(f"Took {time.time() - t_start:.3f} seconds to compute the objective.")
    print("Objective value:", objective_value)
    assert objective_value == pytest.approx(11.005338768207627, 1e-4)

def test_geometry_generator():
    geometry_config = np.loadtxt("data/geometry.txt", dtype=int)
    geometry_ground_truth = geometry_config[4:]
    geo_nx, geo_ny, geo_nz, _ = geometry_config[:4]
    generated_geometry = dda_model._generate_geometry(geo_nx, geo_ny, geo_nz)
    differences = (generated_geometry - geometry_ground_truth)**2
    assert differences.sum() == pytest.approx(0.0, 1e-6)

def test_parameter_setter_2d():
    model = _construct_test_model()
    unique_parameters = model.parameters  # Calls the getter and 2D conversion.
    assert len(unique_parameters.shape) == 2
    assert list(unique_parameters.shape) == list(model.parameters_shape)
    # Manipulate the parameters.
    new_parameters = np.zeros_like(unique_parameters)
    # Set the parameters to test the setter (2D to flat conversion).
    model.parameters = new_parameters
    # Re-get the parameters.
    differences = (model.parameters - new_parameters)**2
    assert differences.sum() == pytest.approx(0.0, 1e-6)

def test_all_parameters_3d():
    model = _construct_test_model()
    all_parameters = model.allParameters()  # Gets extruded 3D parameters.
    assert len(all_parameters.shape) == 3
    assert list(all_parameters.shape) == list(model.domain_shape)

def test_parameters_with_filters():
    model = _construct_test_model()
    unique_parameters = model.parameters  # Calls the getter and 2D conversion.
    assert len(unique_parameters.shape) == 2
    assert list(unique_parameters.shape) == list(model.parameters_shape)
    # Manipulate the parameters.
    new_parameters = np.zeros_like(unique_parameters)
    # Set the parameters to test the setter (2D to flat conversion).
    model.parameters = new_parameters
    # Re-get the parameters.
    differences = (model.parameters - new_parameters)**2
    assert differences.sum() == pytest.approx(0.0, 1e-6)