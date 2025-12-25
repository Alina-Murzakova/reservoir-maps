import numpy as np
import pandas as pd
import logging
import math
import psutil
import os

from scipy.spatial.distance import cdist
from numba import njit, prange
from reservoir_maps.one_phase_model import get_current_So

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_grid(data_volumes):
    """
    Generates a 2D grid of coordinates using shape of input map.
    Args:
        data_volumes: 2D array representing the reservoir map

    Returns:
        np.ndarray: Array of (x, y) grid coordinates.
    """
    # Расчет сетки
    x_coords = np.arange(0, data_volumes.shape[1], 1)
    y_coords = np.arange(0, data_volumes.shape[0], 1)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    return grid_points


def get_saturation_points(row, data_So_init, fluid_params, relative_permeability):
    """
    Calculates initial and current oil saturation at each point of the well trajectory.
    Args:
        row (pd.Series): row of data_wells
        data_So_init (np.ndarray): Initial oil saturation map
        fluid_params (FluidParams): General properties of the reservoir
        relative_permeability (RelativePermeabilityParams): Parameters of the relative permeability curve

    Returns:
        pd.Series: Two lists - initial and current saturation along trajectory.
    """
    traj_x = row['trajectory_x']
    traj_y = row['trajectory_y']
    So_current_wells = []
    So_init_wells = []

    for x, y in zip(traj_x, traj_y):
        # Получаем нефтенасыщенность из карты
        So_init_point = data_So_init[y, x].astype('float32')
        if So_init_point == 0:
            return pd.Series([None, None])

        row_copy = row.copy()
        row_copy['So_init'] = So_init_point

        # Рассчитываем текущую нефтенасыщенность для точки
        So_current_point = get_current_So(row_copy, fluid_params, relative_permeability)

        So_current_wells.append(So_current_point)
        So_init_wells.append(So_init_point)

    return pd.Series([So_init_wells, So_current_wells])


def generate_well_point_vectors(data_wells, map_params, reservoir_params):
    """
    Expands well data to trajectory points
    Args:
        data_wells: (pd.DataFrame): DataFrame with well information and trajectories
        map_params (MapParams): Parameters for accounting in maps
        reservoir_params (reservoir_params): General properties of the reservoir

    Returns:
        Tuple of np.ndarrays with processed well parameters by trajectory points.
    """
    # Параметры для учета АГРП
    sigma_h = None
    l_half_fracture_pixel = None
    if map_params.switch_fracture:
        logger.info("Getting params for accounting auto-fracs at injection wells")
        if reservoir_params.azimuth_sigma_h_min is None or reservoir_params.l_half_fracture is None:
            raise ValueError("For accounting auto-fracs at injection wells, it's necessary to set "
                             "'azimuth_sigma_h_min' и 'l_half_fracture'")
        sigma_h = math.radians(reservoir_params.azimuth_sigma_h_min)
        # Есть несколько вариантов расчета - через закачку и длину
        l_half_fracture_pixel = reservoir_params.l_half_fracture / map_params.size_pixel
    # Списки необходимых параметров
    (x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells, So_init_wells, well_number)\
        = ([], [], [], [], [], [], [], [], [], [], [], [])
    for _, row in data_wells.iterrows():
        x.extend(row['trajectory_x'])
        y.extend(row['trajectory_y'])
        len_trajectory = len(row['trajectory_x'])
        r_eff.extend(
            [row["r_eff"] / map_params.size_pixel] * len_trajectory)  # все в пикселях для снижения размерности
        time_off.extend([row["no_work_time"]] * len_trajectory)
        work_markers.extend([row['work_marker']] * len_trajectory)
        k.extend([row["permeability"]] * len_trajectory)
        h.extend([row['NNT']] * len_trajectory)
        Qo_point = row['Qo_cumsum'] / len_trajectory
        Qo_cumsum.extend([Qo_point] * len_trajectory)
        Winj_point = row['Winj_cumsum'] / len_trajectory
        Winj_cumsum.extend([Winj_point] * len_trajectory)
        So_current_wells.extend(row["So_current"])
        So_init_wells.extend(row["So_init"])
        well_number.extend([row["well_number"]] * len_trajectory)

    x, y = np.array(x), np.array(y)
    well_coord = np.column_stack((x, y))
    So_current_wells, So_init_wells = (np.array(So_current_wells).astype('float32'),
                                       np.array(So_init_wells).astype('float32'))
    r_eff, time_off, work_markers = np.array(r_eff).astype('float32'), np.array(time_off), np.array(work_markers)
    k, h, Qo_cumsum, Winj_cumsum, = np.array(k), np.array(h), np.array(Qo_cumsum), np.array(Winj_cumsum)
    well_number = np.array(well_number)

    return (well_coord, x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells,
            So_init_wells, well_number)


def get_weights(distances, r_eff, k, time_off, delta):
    """
    Computes influence weights from wells on each cell of grid
    Args:
        distances (np.ndarray): Distances from each grid point to each well
        r_eff (np.ndarray): Effective drainage radius
        k (np.ndarray): Reservoir permeability
        time_off (np.ndarray): Time since well was inactive
        delta (float): Coefficient controlling decay rate of well influence

    Returns:
        np.ndarray: Normalized weights of influence.
    """
    # Веса
    psi = np.exp(-delta * k * time_off)
    weights = r_eff * psi / (distances ** 2 + 1e-12)
    weights /= (np.sum(weights, axis=1, keepdims=True) + 1e-12)  # Нормировка веса
    return weights.astype('float32')


def check_memory(matrix_r_ij, max_memory_gb):
    """
    Check whether sufficient memory is available to perform the full computation in a single step
    without splitting into batches.
    Args:
        matrix_r_ij: local influence radius matrix r_ij for each point of well to getting shape
        max_memory_gb: maximum allowed memory usage in gigabytes [GB]

    Returns:
        Flag (bool) - Indicates whether sufficient memory is available to perform the full computation in a single step
        without splitting into batches.
    """
    estimated_size_bytes = matrix_r_ij.shape[0] * matrix_r_ij.shape[1] * 4 * 2 * 0.7  # float32 * 2 arrays
    estimated_size_gb = estimated_size_bytes / (1024.0 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024.0 ** 3)
    logger.info(f"Estimated memory required: ~{estimated_size_gb:.2f} GB, \n"
                f"Available RAM: ~{available_ram_gb:.2f} GB (limit: {max_memory_gb} GB)")
    if estimated_size_gb < min(available_ram_gb, max_memory_gb):
        return True
    else:
        logger.info(f"Not enough memory available for fast computation.\n"
                    f"Batch processing will be used.")
        return False


def batch_generator(valid_points, matrix_r_ij, diff_So, well_coord, r_eff, k, time_off,
                    delta, betta, batch_size):
    """
    This generator function yields intermediate matrices used for weights_diff_saturation, influence_matrix
    Args:
        valid_points: Array of (x, y) pixel coordinates of cells with oil (N, 2).
        matrix_r_ij: Local influence radius matrix r_ij for each point of well (N, M).
        diff_So: Array of changes of oil saturation per well (M,).
        well_coord: Array of (x, y) pixel coordinates of wells (M, 2).
        r_eff: Array of effective radii for each well (M,).
        k:  Array of permeability values for each well (M,).
        time_off: Array of downtime (inactive time) for each well (M,).
        delta: Coefficient controlling decay rate of well influence dependent on inactive time and permeability
        betta: Power coefficient for well interference influence
        batch_size: Number of grid cells to process per batch

    Returns:
        Tuple of:
            - weights_diff_saturation (ndarray): Matrix of saturation difference influence weights (batch_size, M).
            - influence_matrix (ndarray): Matrix of exponential influence factors (batch_size, M).

    """
    # Influence weights
    # Accounting for downtime and permeability
    psi = np.exp(-delta * k * time_off)

    for i in range(0, valid_points.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        vp_batch = valid_points[sl]

        # Calculating of distances from each cell to each well
        logger.debug("Calculating of distances from each cell to each well")
        distances = cdist(vp_batch, well_coord).astype('float32')
        logger.debug("Calculating of weights of wells's influence")
        weights = (r_eff * psi) / (distances ** 2 + 1e-12)
        weights /= (np.sum(weights, axis=1, keepdims=True) + 1e-12)  # weight normalization
        weights = weights.astype('float32')

        # Calculation of coefficients for accounting in exponential decay
        weights_diff_saturation = (weights * diff_So)
        influence_matrix = (((distances + matrix_r_ij[sl]) / r_eff) ** betta)

        yield weights_diff_saturation, influence_matrix


def save_batches_to_disk(batch_generator, save_dir):
    """
    Saves generated batches of data (`weights_diff_saturation` and `influence_matrix`) to disk as `.npy` files.
    Args:
        batch_generator: Generator.
        save_dir: Directory path where the batches will be saved.

    """
    for i, (weights_diff_saturation, influence_matrix) in enumerate(batch_generator):
        # Вычисляем размер каждого массива в байтах
        weights_size = weights_diff_saturation.nbytes
        influence_size = influence_matrix.nbytes

        logger.info(f"Saving batch {i}:")
        logger.info(f"weights_diff_saturation size: {format_size(weights_size)}")
        logger.info(f"influence_matrix size: {format_size(influence_size)}")
        logger.info(f"Total size of batches: {format_size(weights_size + influence_size)}")

        np.save(os.path.join(save_dir, f"weights_diff_saturation_{i}.npy"), weights_diff_saturation)
        np.save(os.path.join(save_dir, f"influence_matrix_{i}.npy"), influence_matrix)


def format_size(size_bytes):
    """
    Converts size in bytes to readable format (KB, MB, GB).
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def batch_generator_from_disk(save_dir):
    """
    Loads saved batches of data from disk and yields them one by one.
    Args:
        save_dir: Directory path where batch files are stored.

    Returns:
        Yields:
        Tuple of:
            - weights_diff_saturation (ndarray):  Matrix of saturation difference influence weights.
            - influence_matrix (ndarray): Matrix of exponential influence factors.

    """
    i = 0
    while True:
        weights_diff_saturation = os.path.join(save_dir, f"weights_diff_saturation_{i}.npy")
        influence_matrix = os.path.join(save_dir, f"influence_matrix_{i}.npy")
        if not os.path.exists(weights_diff_saturation):
            break
        yield np.load(weights_diff_saturation), np.load(influence_matrix)
        i += 1


@njit(parallel=True)
def compute_exp_sum(gamma, weights_diff_saturation, influence_matrix):
    """
    Computes the exponential weighted sum  of saturation differences  for each cell using  Numba for parallel execution.
    Args:
        gamma: Optimization parameter affecting saturation decrease.
        weights_diff_saturation: Matrix of saturation difference influence weights.
        influence_matrix: Matrix of exponential influence factors.

    Returns:
        result (ndarray): Computed weighted sum for each cell.

    """
    n = weights_diff_saturation.shape[0]
    result = np.zeros(n, dtype=np.float32)
    for i in prange(n):
        s = 0.0
        for j in range(weights_diff_saturation.shape[1]):
            s += weights_diff_saturation[i, j] * np.exp(-gamma * influence_matrix[i, j])
        result[i] = s
    return result
