import numpy as np
import logging
from scipy.spatial.distance import cdist

from .one_phase_model import get_f_w
from .input import FluidParams, RelativePermeabilityParams
from .data_processing import get_grid, get_weights


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_water_cut(data_So_current: np.ndarray,
                        fluid_params: FluidParams,
                        relative_permeability: RelativePermeabilityParams,
                        data_wells=None,
                        map_params=None,
                        options=None,
                        initial_oil_saturation=None,
                        ) -> np.ndarray:
    """
    Calculates the current water cut distribution across the grid (map).
    Args:
        data_So_current: Current oil saturation 2D array
        fluid_params: Parameters of the fluids (oil, water)
        relative_permeability: Parameters of the relative permeability curve
        data_wells: Optional well data with adapted Corey parameters and point diagnostics.
        map_params: Optional map-grid parameters required for local correction.
        options: Optional settings controlling weights, batch size and taper shape.
        initial_oil_saturation: Optional initial oil saturation map used to define valid cells.

    Returns:
        2D array of the current water cut (map)
    """
    Sw = 1.0 - data_So_current
    vectorized_get_f_w = np.vectorize(get_f_w)
    f_w_array = vectorized_get_f_w(fluid_params.mu_w, fluid_params.mu_o, fluid_params.Bo, fluid_params.Bw,
                                   Sw, relative_permeability.Fw, relative_permeability.m1, relative_permeability.Fo,
                                   relative_permeability.m2, relative_permeability.Swc, relative_permeability.Sor)
    if (
        data_wells is None
        or map_params is None
        or options is None
        or initial_oil_saturation is None
        or "OFP_adapted" not in data_wells
        or not data_wells["OFP_adapted"].any()
    ):
        return f_w_array

    return _apply_local_ofp_correction(
        f_w_array,
        data_wells,
        map_params,
        options,
        initial_oil_saturation,
        fluid_params,
    )


def _apply_local_ofp_correction(
    base_water_cut,
    data_wells,
    map_params,
    options,
    initial_oil_saturation,
    fluid_params,
):
    """
    Blend adapted well-point water cut into the base map inside effective radii.

    Args:
        base_water_cut: Water-cut map calculated with the base Corey parameters.
        data_wells: Well data with adapted Corey parameters and trajectory diagnostics.
        map_params: Map-grid parameters, including pixel size.
        options: Calculation options controlling weights, batch size and taper power.
        initial_oil_saturation: Initial oil saturation map used to mask invalid cells.
        fluid_params: Oil and water PVT parameters.

    Returns:
        np.ndarray: Water-cut map with local corrections from adapted wells.
    """
    # 1. Разворачиваем данные скважин в векторы точек траекторий.
    #    Для каждой точки сохраняем её радиус влияния, исходный вес и локальную ОФП.
    x = []
    y = []
    r_eff = []
    permeability = []
    time_off = []
    weight_multiplier = []
    adapted_point = []
    adapted_fw = []

    for _, row in data_wells.iterrows():
        length = len(row.trajectory_x)
        x.extend(row.trajectory_x)
        y.extend(row.trajectory_y)
        r_eff.extend([row.r_eff / map_params.size_pixel] * length)
        permeability.extend([row.permeability] * length)
        time_off.extend([row.no_work_time] * length)
        point_multipliers = row.weight_multiplier
        if not isinstance(point_multipliers, (list, tuple, np.ndarray)):
            point_multipliers = [1.0] * length
        weight_multiplier.extend(point_multipliers)

        # Локальная поправка создаётся только точками адаптированных добывающих скважин.
        is_adapted = bool(row.OFP_adapted) and row.work_marker == "prod"
        adapted_point.extend([is_adapted] * length)
        if is_adapted:
            adapted_fw.extend(
                [
                    get_f_w(
                        fluid_params.mu_w,
                        fluid_params.mu_o,
                        fluid_params.Bo,
                        fluid_params.Bw,
                        1.0 - point_so,
                        row.Fw_adapted,
                        row.m1_adapted,
                        row.Fo_adapted,
                        row.m2_adapted,
                        row.Swc_adapted,
                        row.Sor_adapted,
                    )
                    for point_so in row.So_current
                ]
            )
        else:
            adapted_fw.extend([np.nan] * length)

    x = np.asarray(x)
    y = np.asarray(y)
    well_coord = np.column_stack((x, y))
    r_eff = np.asarray(r_eff, dtype="float32")
    permeability = np.asarray(permeability, dtype="float32")
    time_off = np.asarray(time_off, dtype="float32")
    weight_multiplier = np.asarray(weight_multiplier, dtype="float32")
    adapted_point = np.asarray(adapted_point, dtype=bool)
    adapted_fw = np.asarray(adapted_fw, dtype="float32")
    # 2. Выбираем точки, для которых можно физически построить локальную зону влияния.
    local_idx = np.flatnonzero(adapted_point & np.isfinite(adapted_fw) & (r_eff > 0))

    if local_idx.size == 0:
        return base_water_cut

    # 3. Корректируем только валидные ячейки залежи; расчёт ведём пакетами,
    #    чтобы не хранить полную матрицу расстояний в оперативной памяти.
    flat_initial = initial_oil_saturation.ravel()
    valid_mask = flat_initial > 0
    valid_points = get_grid(initial_oil_saturation)[valid_mask]
    base_valid = np.asarray(base_water_cut, dtype="float64").ravel()[valid_mask]
    corrected_valid = base_valid.copy()
    affected_cells = 0

    for start in range(0, len(valid_points), options.batch_size):
        stop = min(start + options.batch_size, len(valid_points))
        points = valid_points[start:stop]
        distances = cdist(points, well_coord).astype("float32")
        weights = get_weights(
            distances,
            r_eff,
            permeability,
            time_off,
            options.delta,
            weight_multiplier,
        )[:, local_idx]
        local_distances = distances[:, local_idx]
        # 4. Вес точки дополнительно гасится косинусным окном:
        #    поправка максимальна на стволе и плавно обращается в ноль на r_eff.
        normalized_distance = np.clip(local_distances / r_eff[local_idx], 0.0, 1.0)
        taper = 0.5 * (1.0 + np.cos(np.pi * normalized_distance))
        taper[local_distances > r_eff[local_idx]] = 0.0
        taper **= options.water_cut_smooth_power
        local_weights = weights * taper

        # 5. В пересекающихся зонах усредняем поправки по весам, сохраняя
        #    суммарную силу влияния в диапазоне [0, 1].
        alpha_sum = np.sum(local_weights, axis=1)
        influenced = alpha_sum > 0
        if np.any(influenced):
            delta_sum = np.sum(
                local_weights * (adapted_fw[local_idx][None, :] - base_valid[start:stop, None]),
                axis=1,
            )
            correction = np.zeros_like(alpha_sum, dtype="float64")
            correction[influenced] = (
                delta_sum[influenced]
                / alpha_sum[influenced]
                * np.clip(alpha_sum[influenced], 0.0, 1.0)
            )
            corrected_valid[start:stop] += correction
            affected_cells += int(np.count_nonzero(influenced))

    corrected = np.asarray(base_water_cut, dtype="float64").copy().ravel()
    corrected[valid_mask] = np.clip(corrected_valid, 0.0, 100.0)
    corrected = corrected.reshape(base_water_cut.shape)

    # 6. В ячейках, пересечённых стволом, фиксируем точное значение,
    #    рассчитанное по локальной ОФП. Для совпавших точек берём среднее.
    exact_sum = {}
    exact_count = {}
    for point_index in local_idx:
        key = (int(y[point_index]), int(x[point_index]))
        if 0 <= key[0] < corrected.shape[0] and 0 <= key[1] < corrected.shape[1]:
            exact_sum[key] = exact_sum.get(key, 0.0) + float(adapted_fw[point_index])
            exact_count[key] = exact_count.get(key, 0) + 1
    for key, value in exact_sum.items():
        corrected[key] = np.clip(value / exact_count[key], 0.0, 100.0)

    logger.info(
        "Applied adapted relative-permeability correction to water cut: %d wells, %d trajectory points, %d cells",
        int(data_wells.loc[data_wells["OFP_adapted"], "well_number"].nunique()),
        len(local_idx),
        affected_cells,
    )
    return corrected
