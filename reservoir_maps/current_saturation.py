import numpy as np
import pandas as pd
import logging
import math

from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from tempfile import TemporaryDirectory
from reservoir_maps.well_interference import get_matrix_r_ij
from reservoir_maps.data_processing import (get_grid, generate_well_point_vectors, get_weights, get_saturation_points,
                                            check_memory, batch_generator, compute_exp_sum, save_batches_to_disk,
                                            batch_generator_from_disk)
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
    # Filtration of wells without values on map_initial_oil_saturation
    data_wells = data_wells[data_wells['So_init'].notna()].reset_index(drop=True)
    maps.initial_oil_saturation = np.where(np.isclose(maps.initial_oil_saturation, 0, atol=1e-1), 0,
                                           maps.initial_oil_saturation)
    # Граница по извлекаемости с учетом КИН
    So_min = maps.initial_oil_saturation * (1 - reservoir_params.KIN)
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
    (well_coord, x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells, So_init_wells,
     well_number) = generate_well_point_vectors(data_wells, map_params, reservoir_params)

    logger.debug("Calculating of total volume of oil production")
    Qo_sum_V = sum(Qo_cumsum) / fluid_params.pho_surf * fluid_params.Bo

    # Проверка текущей насыщенности в скважинах относительно начальной (проблемы с ОФП, So_init или с water_cut)
    check_error_So(So_init_wells, So_current_wells, well_number)
    # Изменение нефтенасыщенности в точках скважин
    diff_So = (So_init_wells - So_current_wells)

    logger.debug("Getting of matrix_r_ij")
    matrix_r_ij = get_matrix_r_ij(valid_points, well_coord, x, y, work_markers, r_eff, h, Qo_cumsum, Winj_cumsum,
                                  map_params.size_pixel, options.max_distance)

    enough_memory = check_memory(matrix_r_ij, options.max_memory_gb)
    if enough_memory:
        # Расстояние от всех ячеек до всех скважин
        logger.debug("Calculating of distances from each cell to each well")
        distances = cdist(valid_points, well_coord).astype('float32')
        logger.debug("Calculating of weights of wells's influence")
        weights = get_weights(distances, r_eff, k, time_off, options.delta)

        weights_diff_saturation = weights * (So_init_wells - So_current_wells)
        influence_matrix = ((distances + matrix_r_ij) / r_eff) ** options.betta
    else:
        weights_diff_saturation = None
        influence_matrix = None

    logger.info("Searching <optimal_gamma> and getting map <So_current>")
    import time
    start_time = time.time()
    optimal_gamma, data_So_current = optimize_gamma(maps.initial_oil_saturation, So_min, flat_So_init, mask,
                                                    valid_points, weights_diff_saturation, influence_matrix,
                                                    matrix_r_ij, data_volumes, Qo_sum_V, diff_So,
                                                    well_coord, r_eff, k, time_off,
                                                    relative_permeability, options, enough_memory)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"Elapsed time of searching <optimal_gamma>: {elapsed_time}")
    return data_So_current


def interpolate_current_saturation(gamma: float,
                                   flat_So_init: np.ndarray,
                                   mask: np.ndarray,
                                   weights_diff_saturation: np.ndarray,
                                   influence_matrix: np.ndarray,
                                   relative_permeability: RelativePermeabilityParams,
                                   enough_memory: bool,
                                   tmp_dir: str,
                                   ) -> np.ndarray:
    """
    Interpolation of the current oil saturation based on influence weights and wells interaction.
    Args:
        gamma: Optimization parameter affecting saturation decrease.
        flat_So_init: Flattened initial oil saturation array.
        mask: Boolean mask array indicating valid cells with oil.
        weights_diff_saturation: Weighted differences between initial and current saturation at wells.
        influence_matrix: Coefficient matrix for accounting influence and distance between wells.
        relative_permeability: Parameters of the relative permeability curve.
        enough_memory: Indicates whether sufficient memory is available to perform the full computation in a single step
        without splitting into batches.
        tmp_dir: Temporary Directory for saving temporary batches

    Returns:
        np.ndarray: Flattened array of current oil saturation values.
    """
    data_So_current = np.copy(flat_So_init)
    if enough_memory:
        # data_So_current[mask] -= np.sum(weights_diff_saturation * np.exp(-gamma * influence_matrix), axis=1)
        data_So_current[mask] -= compute_exp_sum(gamma, weights_diff_saturation, influence_matrix).astype('float32')
    else:
        generator = batch_generator_from_disk(tmp_dir)
        data_So_current[mask] -= interpolate_saturation_changes_batched(gamma, generator).astype('float32')
    data_So_current[mask] = np.maximum(data_So_current[mask], relative_permeability.Sor)
    return data_So_current


def interpolate_saturation_changes_batched(gamma, batch_generator):
    """
    Calculation of the saturation changes sum by batches:
    sum(weights_diff_saturation * exp(-gamma * influence_matrix), axis=1)
    """
    result = []
    for weights_diff_saturation, influence_matrix in batch_generator:
        # contribution = np.sum(weights_diff_saturation * np.exp(-gamma * influence_matrix), axis=1)
        contribution = compute_exp_sum(gamma, weights_diff_saturation, influence_matrix)
        result.append(contribution)
    return np.concatenate(result)


def oil_production_loss(gamma: float,
                        data_So_init: np.ndarray,
                        So_min: np.ndarray,
                        flat_So_init: np.ndarray,
                        mask: np.ndarray,
                        valid_points: np.ndarray,
                        weights_diff_saturation: np.ndarray,
                        influence_matrix: np.ndarray,
                        matrix_r_ij: np.ndarray,
                        data_volumes: np.ndarray,
                        Qo_sum_V: float,
                        diff_So: np.ndarray,
                        well_coord: np.ndarray,
                        r_eff: np.ndarray,
                        k: np.ndarray,
                        time_off: np.ndarray,
                        relative_permeability: RelativePermeabilityParams,
                        enough_memory: bool,
                        tmp_dir: str,
                        ) -> float:
    """Calculates the squared error loss between estimated oil production volume and actual oil production volume."""
    data_So_current = (interpolate_current_saturation(gamma, flat_So_init, mask,
                                                      weights_diff_saturation, influence_matrix,
                                                      relative_permeability, enough_memory, tmp_dir)
                       .reshape(data_So_init.shape))
    # Фактическая добыча из ячеек = (data_So_init - data_So_current) по объему породы
    oil_extracted = (data_So_init - data_So_current) * data_volumes
    # Где нарушено ограничение
    mask_limit = (data_So_current < So_min)
    # В ячейках, где нарушено ограничение, считаем как будто добыли только до S_Hmin
    oil_extracted[mask_limit] = (data_So_init[mask_limit] - So_min[mask_limit]) * data_volumes[mask_limit]
    logger.debug(f"Oil extracted according to the model: {np.sum(oil_extracted)}")
    return (np.sum(oil_extracted) - Qo_sum_V) ** 2


def intermediate_loss(gamma: float,
                      data_So_init: np.ndarray,
                      So_min: np.ndarray,
                      flat_So_init: np.ndarray,
                      mask: np.ndarray,
                      valid_points: np.ndarray,
                      weights_diff_saturation: np.ndarray,
                      influence_matrix: np.ndarray,
                      matrix_r_ij: np.ndarray,
                      data_volumes: np.ndarray,
                      Qo_sum_V: float,
                      diff_So: np.ndarray,
                      well_coord: np.ndarray,
                      r_eff: np.ndarray,
                      k: np.ndarray,
                      time_off: np.ndarray,
                      relative_permeability: RelativePermeabilityParams,
                      enough_memory: bool,
                      tmp_dir: str,
                      ) -> float:
    """Wrapper for oil production loss that logs intermediate optimization results."""
    loss = oil_production_loss(gamma, data_So_init, So_min, flat_So_init, mask, valid_points,
                               weights_diff_saturation, influence_matrix, matrix_r_ij, data_volumes, Qo_sum_V,
                               diff_So, well_coord, r_eff, k, time_off,
                               relative_permeability, enough_memory, tmp_dir)

    logger.debug(f"gamma={gamma:.4f}, loss={loss:.2e}")
    return loss


def optimize_gamma(data_So_init: np.ndarray,
                   So_min: np.ndarray,
                   flat_So_init: np.ndarray,
                   mask: np.ndarray,
                   valid_points: np.ndarray,
                   weights_diff_saturation: np.ndarray,
                   influence_matrix: np.ndarray,
                   matrix_r_ij: np.ndarray,
                   data_volumes: np.ndarray,
                   Qo_sum_V: float,
                   diff_So: np.ndarray,
                   well_coord: np.ndarray,
                   r_eff: np.ndarray,
                   k: np.ndarray,
                   time_off: np.ndarray,
                   relative_permeability: RelativePermeabilityParams,
                   options: Options,
                   enough_memory: bool,
                   ) -> Tuple[float, np.ndarray]:
    """
    Optimizes the gamma parameter to minimize the oil production loss function.
    Args:
        data_So_init: 2D array/grid of initial oil saturation map.
        So_min: Minimum allowable saturation map accounting for recovery factor.
        flat_So_init: Flattened initial oil saturation array.
        mask: Boolean mask array indicating valid cells with oil.
        valid_points: Array of (x, y) pixel coordinates of cells with oil.
        weights_diff_saturation: Weighted differences between initial and current saturation at wells.
        influence_matrix: Coefficient matrix for accounting influence and distance between wells.
        matrix_r_ij: Local influence radius matrix r_ij for each point of well.
        data_volumes:  Pore volumes of reservoir cells.
        Qo_sum_V:  Total cumulative oil production volume.
        diff_So: Array of changes of oil saturation per well.
        well_coord: Array of (x, y) pixel coordinates of wells.
        r_eff: Array of effective radii for each well.
        k: Array of permeability values for each well.
        time_off: Array of downtime (inactive time) for each well.
        relative_permeability: Parameters of the relative permeability curve.
        options: Additional calculation options.
        enough_memory: Flag (bool) - Indicates whether sufficient memory is available to perform the full computation
        in a single step without splitting into batches.

    Returns:
        tuple:
            float: Optimal gamma value found by the optimizer.
            np.ndarray: Current oil saturation 2D array/grid.
    """
    def run_optimization(tmp_dir: Optional[str] = None,
                         ) -> Tuple[float, np.ndarray]:
        res = minimize_scalar(lambda gamma: intermediate_loss(
            gamma, data_So_init, So_min, flat_So_init, mask, valid_points,
            weights_diff_saturation, influence_matrix, matrix_r_ij, data_volumes, Qo_sum_V,
            diff_So, well_coord, r_eff, k, time_off, relative_permeability, enough_memory, tmp_dir),
                              bounds=(0, 1), method='bounded')
        if not res.success:
            logger.warning(f"Gamma optimization did not converge: {res.message}")
        optimal_gamma = res.x
        logger.info(f"Optimal value of gamma: {optimal_gamma:.6f}")
        data_So_current = (interpolate_current_saturation(optimal_gamma, flat_So_init, mask,
                                                          weights_diff_saturation, influence_matrix,
                                                          relative_permeability, enough_memory, tmp_dir)
                           .reshape(data_So_init.shape))
        logger.debug(
            f"Loss: {(np.sum((data_So_init - data_So_current) * data_volumes) - Qo_sum_V) ** 2}")

        return optimal_gamma, data_So_current

    if enough_memory:
        return run_optimization()
    else:
        with TemporaryDirectory() as tmp_dir:
            # Saving temporary batches to temporary directory
            logger.info("Generating and saving batches with intermediate matrices used for weights_diff_saturation, "
                        "influence_matrix")
            save_batches_to_disk(batch_generator(valid_points, matrix_r_ij, diff_So, well_coord, r_eff, k, time_off,
                                                 options.delta, options.betta, options.batch_size), save_dir=tmp_dir)
            return run_optimization(tmp_dir)


def check_error_So(So_init_wells, So_current_wells, well_number):
    """
    Errors of current S_oil because of wrong relative phase permeability, initial oil saturation or water cut of wells
    """
    relative_tolerance = (So_current_wells - So_init_wells) / (So_init_wells + 1e-12)
    absolute_tolerance = (So_current_wells - So_init_wells)
    mask_So_error = (So_current_wells > So_init_wells)

    # Statistics
    count_error_points = np.sum(mask_So_error)
    logger.info(f"Total number of well points where current S_oil > initial S_oil: {count_error_points}")
    if count_error_points > 0:
        mean_relative_error = np.mean(relative_tolerance[mask_So_error])
        mean_absolute_error = np.mean(absolute_tolerance[mask_So_error])
        wells_wrong_So_wells = len(np.unique(well_number[mask_So_error]))
        all_wells = len(np.unique(well_number))
        logger.info(f"Share of wells with at least one point where current S_oil > initial S_oil: "
                    f"{wells_wrong_So_wells / all_wells:.1%}")
        logger.info(f"Mean relative error of S_oil: {mean_relative_error:.1%}")
        logger.info(f"Mean absolute error of S_oil: {mean_absolute_error:.1}")
    return
