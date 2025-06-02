import numpy as np
import pandas as pd
import logging
import math

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
    (x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells, So_init_wells) = (
        [], [], [], [], [], [], [], [], [], [], [])
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

    x, y = np.array(x), np.array(y)
    well_coord = np.column_stack((x, y))
    So_current_wells, So_init_wells = (np.array(So_current_wells).astype('float32'),
                                       np.array(So_init_wells).astype('float32'))
    r_eff, time_off, work_markers = np.array(r_eff).astype('float32'), np.array(time_off), np.array(work_markers)
    k, h, Qo_cumsum, Winj_cumsum, = np.array(k), np.array(h), np.array(Qo_cumsum), np.array(Winj_cumsum)

    return well_coord, x, y, r_eff, time_off, work_markers, k, h, Qo_cumsum, Winj_cumsum, So_current_wells, So_init_wells


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
