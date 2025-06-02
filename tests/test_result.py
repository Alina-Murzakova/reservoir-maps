import pytest
import numpy as np
from reservoir_maps.result import get_maps, ResultMaps


def test_get_maps():
    dict_maps = {"NNT": np.ones((10, 10)), "initial_oil_saturation": np.ones((10, 10)), "porosity": np.ones((10, 10))}
    dict_data_wells = {
        'well_number': [1, 2],
        'work_marker': ['prod', 'inj'],
        'no_work_time': [0.0, 10.0],
        'Qo_cumsum': [1000, 0],
        'Winj_cumsum': [0, 2000],
        'water_cut': [10, 0.0],
        'r_eff': [100, 120],
        'NNT': [5.0, 6],
        'permeability': [20, 15.0],
        'T1_x_pix': [1, 6],
        'T1_y_pix': [1, 6],
        'T3_x_pix': [5, 9],
        'T3_y_pix': [2, 6],
    }
    dict_map_params = {"size_pixel": 50, "switch_fracture": False}
    dict_reservoir_params = {"KIN": 0.25, "azimuth_sigma_h_min": 30, "l_half_fracture": 100}
    dict_fluid_params = {"pho_surf": 0.850, "mu_o": 0.75, "mu_w": 0.3, "Bo": 1.2, "Bw": 1.0}
    dict_relative_permeability = {"Sor": 0.3, "Swc": 0.2, "Fw": 0.3, "m1": 1, "Fo": 1, "m2": 1.0}
    dict_options = {}

    result = get_maps(dict_maps, dict_data_wells, dict_map_params,
                      dict_reservoir_params, dict_fluid_params,
                      dict_relative_permeability)

    assert isinstance(result, ResultMaps)
    assert isinstance(result.data_So_current, np.ndarray)
    assert result.data_So_current.shape == (10, 10)
    assert isinstance(result.data_water_cut, np.ndarray)
    assert result.data_So_current[result.data_So_current > 0.0].min() >= dict_relative_permeability['Sor']



