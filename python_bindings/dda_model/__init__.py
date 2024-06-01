from ._dda_model import DDAModel

import itertools
import numpy as np
from scipy.special import sici


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


class DDAModelWrapper:
    def __init__(
        self,
        num_pixels_xyz: list[int],
        pixel_size_nm: float,
        initial_parameter_values,
        dielectric_constants: list[complex],
        light_direction: list[float],
        light_polarization: list[float],
        light_wavelength_nm: float,
        light_amplitude: float = 1.0,
        symmetry_axes: list[float] = [],
        symmetry_method: str = "4fold",
        symmetry_is_periodic: bool = True,
        integral_power: int = 2,
        integral_xbounds: list[float] | None = None,
        integral_ybounds: list[float] | None = None,
        integral_zbounds: list[float] | None = None,
        background_refractive_index: float | None = None,
        objective_name: str = "IntegratedE",
        apc_max_m: int = 50,
        apc_max_n: int = 50,
        apc_l_m: int | None = None,
        apc_l_n: int | None = None,
        apc_method_name: str = "FCD",
        verbose: bool = False,
        ):
        """
        Constructs a DDAModel.

        Arguments:
            num_pixels_xyz: Shape of the geometry, in pixels, formatted as
                [num_x, num_y, num_z].
            pixel_size_nm: Increment length of one side of each pixel / voxel,
                in nanometers.
            initial_parameter_values: Initial values for the parameters. Due to
                how the C++ library initializes parameters, this needs to be
                3 * num_pixels with repeated values (e.g., the first 3 values
                correspond to the pixel at [0,0,0]).
            dielectric_constants: A list of length two, containing the complex
                dielectric constant of the background material and the
                foreground material.
            light_direction: A 3d unit vector describing the angle of the
                incident light wave.
            light_polarization: A 3d unit vector describing the polarization of
                the incident light wave.
            light_wavelength_nm: The wavelength of the light in nanometers.
            light_amplitude: The amplitude of the light.
            symmetry_axes: The (x,y) coordinates of the lines along which to
                mirror the geometry, if the symmetry_method is "4fold"
            symmetry_method: Either "4fold" or "None".
            symmetry_is_periodic: If true, the structure is assumed to repeat
                in a periodic way.
            integral_power: The objective function maximizes the Lp functional
                norm of the field, where integral_power is the value of p. For
                the standard Euclidean norm (power), use 2.
            integral_xbounds: The x-axis bounds over which to calculate the
                power integral for the objective. Defaults to the geometry.
            integral_ybounds: The y-axis bounds over which to calculate the
                power integral for the objective. Defaults to the geometry.
            integral_zbounds: The z-axis bounds over which to calculate the
                power integral for the objective. Defaults to the geometry.
            background_refractive_index: Refractive index of the background.
                Defaults to use the background material listed in
                dielectric_materials.
            objective_name: Choice for the objective. Options are "IntegratedE"
                and "PointE".
            apc_max_m: MAXm numerical parameter for A-product-core. 
            apc_max_n: MAXn numerical parameter for A-product-core.
            apc_l_m: Lm numerical parameter for A-product-core. Defaults to
                the number of pixels in the y-axis.
            apc_l_n: Ln numerical parameter for A-product-core. Defaults to
                the number of pixels in the x-axis.
            apc_method_name: Choice of method used to compute the A-product.
            verbose: Whether to print logging information from C++.
        """
        if len(num_pixels_xyz) != 3:
            raise ValueError("num_pixels_xyz must contain the number of pixels "
                            f"in [x, y, z] but instead found {num_pixels_xyz}.")
        if len(symmetry_axes) != 2 and symmetry_method == "4fold":
            raise ValueError("If symmetry_method is 4fold, symmetry_axes must "
                            f"be 2-dimensional. Instead found {symmetry_axes}.")
        valid_methods = ["4fold", "none"]
        if symmetry_method not in valid_methods:
            raise ValueError("Unsupported symmetry method "
                             f"{self._symmetry_method}. Expected a method from "
                             f"{valid_methods}.")
        if symmetry_method == "4fold":
            if num_pixels_xyz[0] % 2 != 0:
                raise ValueError("For 4fold symmetry, num_x must be even.")
            if num_pixels_xyz[1] % 2 != 0:
                raise ValueError("For 4fold symmetry, num_y must be even.")
        # Input checking is done, initialize the parameters.
        self._domain_shape = num_pixels_xyz
        self._parameter_shape = num_pixels_xyz[:2] # XY only.
        if symmetry_method == "4fold":
            self._parameter_shape = [
                num_pixels_xyz[0] // 2, num_pixels_xyz[1] // 2
            ]
        self._symmetry_method = symmetry_method
        self._pixel_dimensions = num_pixels_xyz
        num_pixels_total = np.prod(self._pixel_dimensions)
        num_x, num_y, num_z = self._domain_shape
        geometry = _generate_geometry(num_x, num_y, num_z)
        # Objective configuration.
        if not integral_xbounds:
            integral_xbounds = [0.0, float(num_x) - 1]
        if not integral_ybounds:
            integral_ybounds = [0.0, float(num_y) - 1]
        if not integral_zbounds:
            integral_zbounds = [0.0, float(num_z) - 1]
        objective_config = [integral_power]
        objective_config += integral_xbounds
        objective_config += integral_ybounds
        objective_config += integral_zbounds
        objective_config += [0.95, 50.0]  # Unused filtering defaults.
        # Unused filtering default values.
        filter_beta_min = 0.0
        filter_beta_max = 50.0
        filter_ita = 0.5
        filter_method = "piecewise"
        filter_iterations = [100]
        filter_radii = [2.0]
        filter_enable = False
        # TODO: Verify that this is the correct default behavior. Also check
        # that max_m and max_n should not default differently (e.g., to 2*l_m).
        if not apc_l_n:
            apc_l_n = num_x
        if not apc_l_m:
            apc_l_m = num_y
        # Cached integral values.
        sici_delta = 0.1
        sici_n = 1_000_000
        integral_positions = np.linspace(
            sici_delta, sici_delta * sici_n, sici_n - 1)
        si, ci = sici(integral_positions)
        # Calculate background refractive index from the dielectric materials.
        if not background_refractive_index:
            background_refractive_index = np.sqrt(
                np.real(dielectric_constants[0])
            )
        # Construct the model.
        self._model = DDAModel(
            filter_beta_min, filter_beta_max, filter_ita, filter_method,
            filter_iterations, filter_radii, filter_enable,
            self._symmetry_method, symmetry_axes, symmetry_is_periodic,
            objective_name, objective_config,
            geometry, initial_parameter_values,
            num_x, num_y, num_z, num_pixels_total, 
            light_direction, light_amplitude, light_polarization,
            light_wavelength_nm, dielectric_constants,
            background_refractive_index,
            apc_max_m, apc_max_n, apc_l_m, apc_l_n,
            apc_method_name, pixel_size_nm,
            si, ci, sici_delta, verbose,
        )

    def _flat_to_2d(self, x):
        """Convert flattened parameters from C++ into 2D tensors.

        Note: The C++ API expects row-major (C-order) inputs, so the flatten
        and reshape commands must obey this format.
        """
        return np.reshape(x, self._parameter_shape, order="C")

    def _2d_to_flat(self, x):
        """Flatten 2D parameters into a format ready to pass to the C++."""
        return x.flatten()

    def _flat_to_3d(self, x):
        """Converts flattened C++ parameters into a 3D parameters."""
        return np.reshape(x, self._domain_shape, order="C")

    def objective(
        self,
        bgs_max_iter: int = 100_000,
        bgs_max_error: float = 1e-5,
        ):
        num_pixels_total = np.prod(self._pixel_dimensions)
        origin_polarization = [0 + 0j] * 3 * num_pixels_total
        self._model.solveElectricField(
            origin_polarization, 
            bgs_max_iter,
            bgs_max_error,
        )
        return self._model.calculateObjective()

    def gradients(
        self,
        current_objective_value: float,
        finite_difference_epsilon: float = 0.001,
        bgs_max_iter: int = 100_000,
        bgs_max_error: float = 1e-5,
        ):
        grads = self._model.calculateGradients(
            finite_difference_epsilon,
            current_objective_value,
            bgs_max_iter,
            bgs_max_error,
        )
        return self._flat_to_2d(grads)

    @property
    def domain_shape(self):
        """Returns the shape of the input geometry (the domain)."""
        return self._domain_shape

    @property
    def parameters_shape(self):
        """Returns the shape of the unique learnable parameters."""
        return self._parameter_shape

    @property
    def parameters(self):
        """Returns the unique, learnable parameters."""
        p = self._model.getParameters()
        return self._flat_to_2d(p)

    def allParameters(self):
        """Returns the reflected, extruded version of the parameters."""
        p = self._model.getDielectrics()
        p = p[::3]  # TODO: Fix format from the C++ module to not 3x duplicate.
        return self._flat_to_3d(p)
    
    def getElectricField(self):
        """Returns the electric field for a given set of parameters."""
        return self._model.getElectricField()

    @parameters.setter
    def parameters(self, value):
        """Sets unique parameters. Sets a 2D array of shape [num_x, num_y]."""
        value = self._2d_to_flat(value)
        self._model.setParameters(value)
