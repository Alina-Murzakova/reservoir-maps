import numpy as np
import pandas as pd

from reservoir_maps.current_saturation import oil_production_loss
from reservoir_maps.input import (
    FluidParams,
    MapParams,
    Options,
    RelativePermeabilityParams,
    ReservoirParams,
)
from reservoir_maps.ofp_adaptation import (
    adapt_local_relative_permeability,
    calculate_weight_multipliers,
)
from reservoir_maps.one_phase_model import get_f_w
from reservoir_maps.water_cut import calculate_water_cut
from reservoir_maps.data_processing import get_weights, batch_generator


def _adapt_one_bad_point():
    data_wells = pd.DataFrame(
        {
            "well_number": ["P-1"],
            "work_marker": ["prod"],
            "Qo_cumsum": [1_000.0],
            "water_cut": [10.0],
            "r_eff": [100.0],
            "permeability": [20.0],
            "no_work_time": [0.0],
            "trajectory_x": [[1]],
            "trajectory_y": [[1]],
            "So_init": [[0.5]],
            "So_current": [[0.7]],
        }
    )
    fluid = FluidParams(pho_surf=0.85, mu_o=0.75, mu_w=0.3, Bo=1.2, Bw=1.0)
    relative = RelativePermeabilityParams(Sor=0.3, Swc=0.2, Fw=0.3, m1=1.0, Fo=1.0, m2=1.0)
    reservoir = ReservoirParams(KIN=0.3)
    options = Options()
    multipliers, _ = calculate_weight_multipliers(
        np.array([0.5]), np.array([0.7]), relative.Sor, options.alpha_error_points,
        options.min_weight_multiplier,
    )
    adapted, data_wells = adapt_local_relative_permeability(
        data_wells,
        np.array([0.5]),
        np.array([0.7]),
        np.array(["prod"]),
        np.array(["P-1"]),
        multipliers,
        fluid,
        relative,
        reservoir,
        options,
    )
    return adapted, data_wells, fluid, relative, options


def test_bad_point_weight_is_reduced_by_severity():
    multipliers, bad_mask = calculate_weight_multipliers(
        np.array([0.5, 0.6]),
        np.array([0.6, 0.5]),
        Sor=0.3,
        alpha=2.0,
        min_multiplier=0.2,
    )
    assert bad_mask.tolist() == [True, False]
    assert np.isclose(multipliers[0], np.exp(-1.0))
    assert multipliers[1] == 1.0


def test_weight_reduction_is_used_in_full_and_batch_paths():
    distances = np.array([[1.0, 1.0]], dtype="float32")
    multipliers = np.array([0.2, 1.0], dtype="float32")
    expected = get_weights(
        distances,
        np.ones(2),
        np.ones(2),
        np.zeros(2),
        delta=0.0001,
        weight_multipliers=multipliers,
    )
    generated, _ = next(
        batch_generator(
            valid_points=np.array([[0.0, 0.0]]),
            matrix_r_ij=np.zeros((1, 2)),
            diff_So=np.ones(2),
            well_coord=np.array([[1.0, 0.0], [-1.0, 0.0]]),
            r_eff=np.ones(2),
            k=np.ones(2),
            time_off=np.zeros(2),
            delta=0.0001,
            betta=2.0,
            batch_size=1,
            weight_multipliers=multipliers,
        )
    )
    assert np.allclose(expected, generated)


def test_local_ofp_adaptation_returns_parameters_and_valid_saturation():
    adapted, data_wells, _, _, _ = _adapt_one_bad_point()
    result = data_wells.attrs["adapted_relative_permeability"]

    assert adapted[0] <= 0.5
    assert data_wells.loc[0, "OFP_adapted"]
    assert result["well_number"].tolist() == ["P-1"]
    assert result.loc[0, "Fw"] < 0.3
    assert result.loc[0, "eps_so_mean"] > 0


def test_water_cut_uses_adapted_ofp_at_well_cell():
    adapted, data_wells, fluid, relative, options = _adapt_one_bad_point()
    saturation = np.full((3, 3), adapted[0])
    water_cut = calculate_water_cut(
        saturation,
        fluid,
        relative,
        data_wells=data_wells,
        map_params=MapParams(size_pixel=50),
        options=options,
        initial_oil_saturation=np.full((3, 3), 0.5),
    )
    expected = get_f_w(
        fluid.mu_w,
        fluid.mu_o,
        fluid.Bo,
        fluid.Bw,
        1.0 - adapted[0],
        data_wells.loc[0, "Fw_adapted"],
        data_wells.loc[0, "m1_adapted"],
        data_wells.loc[0, "Fo_adapted"],
        data_wells.loc[0, "m2_adapted"],
        data_wells.loc[0, "Swc_adapted"],
        data_wells.loc[0, "Sor_adapted"],
    )
    assert np.isclose(water_cut[1, 1], expected)


def test_material_balance_loss_penalizes_rrr_above_irr():
    relative = RelativePermeabilityParams(Sor=0.2, Swc=0.2, Fw=0.3, m1=1.0, Fo=1.0, m2=1.0)
    kwargs = dict(
        gamma=0.0,
        data_So_init=np.array([[0.5]]),
        So_min=np.array([[0.2]]),
        flat_So_init=np.array([0.5]),
        mask=np.array([True]),
        valid_points=np.array([[0, 0]]),
        weights_diff_saturation=np.array([[-0.1]], dtype="float32"),
        influence_matrix=np.array([[0.0]], dtype="float32"),
        matrix_r_ij=np.array([[0.0]], dtype="float32"),
        data_volumes=np.array([[1.0]]),
        Qo_sum_V=0.1,
        diff_So=np.array([-0.1]),
        well_coord=np.array([[0, 0]]),
        r_eff=np.array([1.0]),
        k=np.array([1.0]),
        time_off=np.array([0.0]),
        relative_permeability=relative,
        enough_memory=True,
        tmp_dir=None,
    )
    unpenalized = oil_production_loss(**kwargs, penalty_weight=0.0)
    penalized = oil_production_loss(**kwargs, penalty_weight=100.0)
    assert penalized > unpenalized
