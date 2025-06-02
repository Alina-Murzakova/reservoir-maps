import pytest
from reservoir_maps.input import dataclass_from_dict, validate_and_prepare_data_wells
from reservoir_maps.input import MapParams, FluidParams, ReservoirParams, RelativePermeabilityParams


@pytest.mark.parametrize("cls, data", [
    (MapParams, {"size_pixel": 50, "switch_fracture": False}),
    (ReservoirParams, {"KIN": 0.25, "azimuth_sigma_h_min": 30, "l_half_fracture": 100}),
    (FluidParams, {"pho_surf": 0.850, "mu_o": 0.75, "mu_w": 0.3, "Bo": 1.2, "Bw": 1.0}),
    (RelativePermeabilityParams, {"Sor": 0.2, "Swc": 0.2, "Fw": 0.3, "m1": 1, "Fo": 1, "m2": 1.0}),
])
def test_dataclass_from_dict_params(cls, data):
    result = dataclass_from_dict(cls, data)
    assert isinstance(result, cls)
    for key, value in data.items():
        assert getattr(result, key) == value


def test_reservoir_params():
    res_params = ReservoirParams(KIN=0.5, azimuth_sigma_h_min=30, l_half_fracture=15.0)
    assert isinstance(res_params.KIN, float)


def test_invalid_dict():
    data = {"mu_o": 0.75}
    with pytest.raises(ValueError):
        dataclass_from_dict(FluidParams, data)


def test_KIN_range():
    with pytest.raises(ValueError):
        ReservoirParams(KIN=1.5)


# Пример валидных данных
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
    'T1_x_pix': [10, 20],
    'T1_y_pix': [10, 20],
    'T3_x_pix': [15, 25],
    'T3_y_pix': [10, 25],
}


def test_validate_and_prepare_data_wells():
    df = validate_and_prepare_data_wells(dict_data_wells)
    assert len(dict_data_wells['well_number']) == len(dict_data_wells['no_work_time'])
    assert 'permeability' in df.columns
    assert 'trajectory_y' in df.columns


def test_work_marker():
    invalid = dict_data_wells.copy()
    invalid['work_marker'] = ['нагн', 'prod']
    with pytest.raises(ValueError):
        validate_and_prepare_data_wells(invalid)


def test_missing_key():
    invalid = dict_data_wells.copy()
    del invalid['Qo_cumsum']
    with pytest.raises(ValueError):
        validate_and_prepare_data_wells(invalid)
