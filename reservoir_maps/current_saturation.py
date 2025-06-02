import numpy as np
import pandas as pd
import logging
import math

from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from reservoir_maps.well_interference import get_matrix_r_ij
from reservoir_maps.data_processing import get_grid, generate_well_point_vectors, get_weights, get_saturation_points
from .input import (MapParams, ReservoirParams, FluidParams, RelativePermeabilityParams, Options, MapCollection)
from .utils import update_injection_trajectory

logger = logging.getLogger(__name__)


def calculate_current_saturation(maps: MapCollection,
                                 data_wells: pd.DataFrame,
                                 map_params: MapParams,
                                 reservoir_params: ReservoirParams,
                                 fluid_params: FluidParams,
                                 relative_permeability: RelativePermeabilityParams,
                                 options: Options) -> np.ndarray:
    """
    Calculates the current oil saturation distribution across the grid (map).
    Args:
        maps: Collection of input maps
        data_wells: DataFrame containing well data
        map_params: Parameters for accounting in maps
        reservoir_params: General properties of the reservoir
        fluid_params: Parameters of the fluids (oil, water)
        relative_permeability: Parameters of the relative permeability curve
        options: Additional calculation options

    Returns:
        2D array of the current oil saturation distributed across the grid
    """
    logger.debug("Accounting auto-fracs at injection wells")
    if map_params.switch_fracture:
        # Getting params for accounting auto-fracs at injection wells
        sigma_h = math.radians(reservoir_params.azimuth_sigma_h_min)
        l_half_fracture_pixel = reservoir_params.l_half_fracture / map_params.size_pixel
        data_wells = data_wells.apply(update_injection_trajectory, args=(sigma_h, l_half_fracture_pixel), axis=1)

    logger.debug("Get values of saturations for each wells trajectory point")
    data_wells[['So_init', 'So_current']] = data_wells.apply(get_saturation_points,
                                                             args=(maps.initial_oil_saturation,
                                                                   fluid_params,
                                                                   relative_permeability,
                                                                   ), axis=1)
    maps.initial_oil_saturation = np.where(maps.initial_oil_saturation < relative_permeability.Sor,
                                           0, maps.initial_oil_saturation)
    # Расчет порового объема
    logger.debug("Calculating reservoir pore volume <data_volumes>")
    data_volumes = map_params.size_pixel ** 2 * maps.NNT * maps.porosity

    logger.debug("Getting <grid_points>")
    grid_points = get_grid(data_volumes)

    logger.debug("Getting <flat_So_init>, <mask>, <valid_points>")
    # Преобразование исходной карты нефтенасыщенности в одномерный массив
    flat_So_init = maps.initial_oil_saturation.ravel()
    # Выбор только тех ячеек, где есть нефть
    mask = flat_So_init > 0
    valid_points = grid_points[mask]

    logger.debug("Generating vectors of wells's points")
    well_coord, x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells, So_init_wells = (
        generate_well_point_vectors(data_wells, map_params, reservoir_params))
    # Расстояние от всех ячеек до всех скважин
    logger.debug("Calculating of distances from each cell to each well")
    distances = cdist(valid_points, well_coord).astype('float32')
    logger.debug("Calculating of weights of wells's influence")
    weights = get_weights(distances, r_eff, k, time_off, options.delta)

    weights_diff_saturation = weights * (So_init_wells - So_current_wells)

    logger.debug("Getting of matrix_r_ij")
    matrix_r_ij = get_matrix_r_ij(valid_points, well_coord, x, y, work_markers, r_eff, h, Qo_cumsum, Winj_cumsum,
                                  map_params.size_pixel)
    influence_matrix = ((distances + matrix_r_ij) / r_eff) ** options.betta
    logger.debug("Calculating of total volume of oil production")
    Qo_sum_V = sum(Qo_cumsum) / fluid_params.pho_surf * fluid_params.Bo
    logger.info("Searching <optimal_gamma> and getting map <So_current>")
    # Граница по извлекаемости с учетом КИН
    So_min = maps.initial_oil_saturation * (1 - reservoir_params.KIN)
    optimal_gamma, data_So_current = optimize_gamma(maps.initial_oil_saturation, So_min, flat_So_init, mask,
                                                    weights_diff_saturation,
                                                    influence_matrix, data_volumes, Qo_sum_V, relative_permeability)

    return data_So_current


def interpolate_current_saturation(gamma: float,
                                   flat_So_init: np.ndarray,
                                   mask: np.ndarray,
                                   weights_diff_saturation: np.ndarray,
                                   influence_matrix: np.ndarray,
                                   relative_permeability: RelativePermeabilityParams) -> np.ndarray:
    """
    Interpolation of the current oil saturation based on influence weights and wells interaction.
    Args:
        gamma: Optimization parameter affecting saturation decrease.
        flat_So_init: Flattened initial oil saturation array.
        mask: Boolean mask array indicating valid cells with oil.
        weights_diff_saturation: Weighted differences between initial and current saturation at wells.
        influence_matrix: Coefficient matrix for accounting influence and distance between wells.
        relative_permeability: Parameters of the relative permeability curve.

    Returns:
        np.ndarray: Flattened array of current oil saturation values.
    """
    data_So_current = np.copy(flat_So_init)
    data_So_current[mask] -= np.sum(weights_diff_saturation * np.exp(-gamma * influence_matrix), axis=1)
    data_So_current[mask] = np.maximum(data_So_current[mask], relative_permeability.Sor)
    return data_So_current


def oil_production_loss(gamma: float,
                        data_So_init: np.ndarray,
                        So_min: np.ndarray,
                        flat_So_init: np.ndarray,
                        mask: np.ndarray,
                        weights_diff_saturation: np.ndarray,
                        influence_matrix: np.ndarray,
                        data_volumes: np.ndarray,
                        Qo_sum_V: float,
                        relative_permeability: RelativePermeabilityParams) -> float:
    """Calculates the squared error loss between estimated oil production volume and actual oil production volume."""
    data_So_current = (interpolate_current_saturation(gamma, flat_So_init, mask, weights_diff_saturation,
                                                      influence_matrix, relative_permeability)
                       .reshape(data_So_init.shape))
    # Фактическая добыча из ячеек = (data_So_init - data_So_current) по объему породы
    oil_extracted = (data_So_init - data_So_current) * data_volumes
    # Где нарушено ограничение
    mask_limit = (data_So_current < So_min)
    # В ячейках, где нарушено ограничение, считаем как будто добыли только до S_Hmin
    oil_extracted[mask_limit] = (data_So_init[mask_limit] - So_min[mask_limit]) * data_volumes[mask_limit]

    return (np.sum(oil_extracted) - Qo_sum_V) ** 2


def intermediate_loss(gamma: float,
                      data_So_init: np.ndarray,
                      So_min: np.ndarray,
                      flat_So_init: np.ndarray,
                      mask: np.ndarray,
                      weights_diff_saturation: np.ndarray,
                      influence_matrix: np.ndarray,
                      data_volumes: np.ndarray,
                      Qo_sum_V: float,
                      relative_permeability: RelativePermeabilityParams) -> float:
    """Wrapper for oil production loss that logs intermediate optimization results."""
    loss = oil_production_loss(gamma, data_So_init, So_min, flat_So_init, mask, weights_diff_saturation,
                               influence_matrix, data_volumes, Qo_sum_V, relative_permeability)
    logger.debug(f"gamma={gamma:.4f}, loss={loss:.2e}")
    return loss


def optimize_gamma(data_So_init: np.ndarray,
                   So_min: np.ndarray,
                   flat_So_init: np.ndarray,
                   mask: np.ndarray,
                   weights_diff_saturation: np.ndarray,
                   influence_matrix: np.ndarray,
                   data_volumes: np.ndarray,
                   Qo_sum_V: float,
                   relative_permeability: RelativePermeabilityParams) -> Tuple[float, np.ndarray]:
    """
    Optimizes the gamma parameter to minimize the oil production loss function.
    Args:
        data_So_init: 2D array/grid of initial oil saturation map.
        So_min: Minimum allowable saturation map accounting for recovery factor.
        flat_So_init: Flattened initial oil saturation array.
        mask: Boolean mask array indicating valid cells with oil.
        weights_diff_saturation: Weighted differences between initial and current saturation at wells.
        influence_matrix: Coefficient matrix for accounting influence and distance between wells.
        data_volumes:  Pore volumes of reservoir cells.
        Qo_sum_V:  Total cumulative oil production volume.
        relative_permeability: Parameters of the relative permeability curve.

    Returns:
        tuple:
            float: Optimal gamma value found by the optimizer.
            np.ndarray: Current oil saturation 2D array/grid.
    """
    # """Optimizes the gamma parameter to minimize the oil production loss function."""

    res = minimize_scalar(lambda gamma: intermediate_loss(
        gamma, data_So_init, So_min, flat_So_init, mask, weights_diff_saturation, influence_matrix,
        data_volumes, Qo_sum_V, relative_permeability),
                          bounds=(0, 1), method='bounded')
    if not res.success:
        logger.warning(f"Gamma optimization did not converge: {res.message}")
    optimal_gamma = res.x
    logger.info(f"Optimal value of gamma: {optimal_gamma}")
    data_So_current = interpolate_current_saturation(optimal_gamma, flat_So_init, mask, weights_diff_saturation,
                                                     influence_matrix, relative_permeability).reshape(data_So_init.shape)
    logger.info(
        f"Loss: {(np.sum((data_So_init - data_So_current) * data_volumes) - Qo_sum_V) ** 2}")

    return optimal_gamma, data_So_current
