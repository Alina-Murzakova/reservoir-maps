import numpy as np
from .one_phase_model import get_f_w
from .input import MapCollection, FluidParams, RelativePermeabilityParams


def calculate_water_cut(data_So_current: np.ndarray,
                        fluid_params: FluidParams,
                        relative_permeability: RelativePermeabilityParams,
                        ) -> np.ndarray:
    """
    Calculates the current water cut distribution across the grid (map).
    Args:
        data_So_current: Current oil saturation 2D array
        fluid_params: Parameters of the fluids (oil, water)
        relative_permeability: Parameters of the relative permeability curve
    Returns:
        2D array of the current water cut (map)
    """
    Sw = 1.0 - data_So_current
    vectorized_get_f_w = np.vectorize(get_f_w)
    f_w_array = vectorized_get_f_w(fluid_params.mu_w, fluid_params.mu_o, fluid_params.Bo, fluid_params.Bw,
                                   Sw, relative_permeability.Fw, relative_permeability.m1, relative_permeability.Fo,
                                   relative_permeability.m2, relative_permeability.Swc, relative_permeability.Sor)
    return f_w_array
