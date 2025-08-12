import numpy as np
import math

"""
-----Accounting for Influence Radii in Oil Saturation-----
Module for determining the shape of an asymmetric drainage/injection zone of wells,
where each direction of the vectors alpha_ij is characterized by different radii r_jl.
The calculation of the distorted zone is based on well interference and contribution coefficients.

Algorithm:
1. Calculate direction matrix alpha_ij for each cell of grid.
2. Compute correlation table r_jl(alpha_ij) using calculate_r_jl_values for each point.
3. Let pass matrix alpha_ij through get_r_jl to get radius matrix.
4. Add the resulting array to the general matrix, use for oil saturation calculations and updating effective radius of wells.
"""


def get_matrix_r_ij(valid_points, well_coord, x, y, work_markers, r_eff, h, Qo_cumsum, Winj_cumsum, size_pixel,
                    max_distance):
    """
    Calculate local influence radius matrix r_ij for each point of well.
    Args:
        valid_points: Boolean mask array indicating valid cells with oil.
        well_coord (2-D numpy array): coordinates of wells [[x, y]...] (pixel coords)
        x (numpy array): x - coordinates (pixel coords)
        y (numpy array): y - coordinates (pixel coords)
        work_markers (numpy array): work_markers of wells [(str): 'prod' or 'inj']
        r_eff (numpy array): effective radius of wells [m]
        h (numpy array): work_markers oil-saturated thickness of wells
        Qo_cumsum (numpy array): cumulative oil productions of wells [t]
        Winj_cumsum(numpy array): cumulative fluid injection of wells [m³]
        size_pixel (int): size of one pixel (cell) in the map grid [m]
        max_distance (float): maximum distance for the nearest surrounding (influencing) wells [m]
    Returns:
        2D np.array [grid.shape, len(wells)]: matrix r_ij for each point of well.
    """
    # Расчет локальных матриц взаимодействия для точек
    matrix_r_ij = np.empty((valid_points[:, 1].shape[0], len(x)), dtype=np.float32)
    index = 0
    for index_point in range(len(x)):
        x_point, y_point, work_marker_point, r_eff_n = (x[index_point], y[index_point],
                                                        work_markers[index_point], r_eff[index_point])
        if work_marker_point == 'prod' or work_marker_point == 'inj':
            # Строка из матрицы влияния с созависимыми скважинами
            interference_array, mask_general = calc_interference_matrix(np.array([x_point, y_point]), work_marker_point,
                                                                        well_coord, h, Winj_cumsum, Qo_cumsum,
                                                                        work_markers, max_distance / size_pixel)
            if not np.any(mask_general):
                matrix_r_ij[:, index] = np.full(valid_points[:, 1].shape[0], r_eff_n / size_pixel)
            else:
                # Центры данных зависимых скважин
                centers_x, centers_y = x[mask_general], y[mask_general]
                # Считаем массив направлений alpha_ij
                array_alpha_ij = calculate_alpha((x_point, y_point), (valid_points[:, 0], valid_points[:, 1]))
                # Рассчитаем корреляционную таблицу r_jl(alpha_ij) через функцию calculate_r_jl_values
                angles, r_jl_values_prod_well = calculate_r_jl_values((x_point, y_point), r_eff_n, interference_array,
                                                                      centers_x, centers_y)
                # array_alpha_ij прогоним через функцию get_r_jl - и получим матрицу радиусов для одной точки
                array_r_jl = get_r_jl(array_alpha_ij, angles, r_jl_values_prod_well)
                matrix_r_ij[:, index] = array_r_jl.ravel()
            index += 1
        else:
            matrix_r_ij[:, index] = np.full(valid_points[:, 1].shape[0], r_eff_n / size_pixel)
            index += 1
    return matrix_r_ij


def calc_interference_matrix(point_coord, work_marker_point, grid_point_wells, array_h, array_Winj, array_Qo,
                             array_work_marker, max_distance):
    """
    Compute interference coefficients between wells.
    Args:
        point_coord (numpy array) [x, y]: coordinates of well (pixel coords)
        work_marker_point (str) 'prod' or 'inj': work_marker of well
        grid_point_wells (2-D numpy array): coordinates of wells [[x, y]...] (pixel coords)
        array_h (numpy array): work_markers oil-saturated thickness of wells
        array_Winj (numpy array): cumulative fluid injection of wells [m³]
        array_Qo (numpy array): cumulative oil productions of wells [t]
        array_work_marker (numpy array): work_markers of wells [(str): 'prod' or 'inj']
        max_distance: maximum distance for determining influencing wells [pix]

    Returns:
        List of two np.array: [lambda_ij - interference coefficients between wells,
                                mask_general - boolean mask array indicating influencing wells].
    """
    # Расчет коэффициентов участия и влияния
    array_distance = np.linalg.norm(grid_point_wells - point_coord, axis=1)
    mask_nearest_points = (array_distance <= max_distance) & (array_distance > 0)
    mask_marker = array_work_marker != work_marker_point
    mask_general = mask_nearest_points & mask_marker
    if work_marker_point == 'prod':
        # Расчет коэффициентов влияния для добывающих скважин
        lambda_ij = array_h[mask_general] * array_Winj[mask_general] / array_distance[mask_general]
        lambda_ij = lambda_ij / lambda_ij.sum()
    else:
        # Расчет коэффициентов участия для нагнетательных скважин
        lambda_ij = array_h[mask_general] * array_Qo[mask_general] / array_distance[mask_general]
        lambda_ij = 1 - lambda_ij / lambda_ij.sum()
    return lambda_ij, mask_general


def calculate_alpha(point_center, point_cell):
    """
    Vectorized angle calculation (radians) from 'point_center' to 'point_cell'.
    Args:
        point_center (tuple[float, float]): Center point of the well (x, y)
        point_cell (tuple[np.ndarray, np.ndarray]): Coordinates of grid cells (x_array, y_array).

    Returns:
        np.ndarray: Matrix of angles in range [0, 2π).
    """
    x1, y1 = point_center
    x2, y2 = point_cell  # Распаковываем массивы координат

    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx)
    return np.where(angle < 0, angle + 2 * np.pi, angle)


def calculate_r_jl_values(point_well, r_eff, interferense_values, centers_x, centers_y,
                          delta_theta=np.pi / 10, num_points=365):
    """
    Compute r_jl(alpha) table for point of one well — influence radii depending on angle.
    Args:
        point_well (numpy array) [x, y]: coordinates of well (pixel coords)
        r_eff (int): effective radius of well [m]
        interferense_values (numpy array): interference coefficients between wells
        centers_x (numpy array):  x - coordinates of influencing wells (pixel coords)
        centers_y (numpy array):  y - coordinates of influencing wells (pixel coords)
        delta_theta (int) = np.pi / 10: Angle for calculating the projections of alpha_ij onto the rays lambda_k [rad]
        num_points (int) = 365: Calculation of angles for constructing a correlation table of interaction radii
                                based on partitioning the [0, 2π) space into num_points intervals
    Returns:
        List of two np.array: [alpha_angles, r_jl_values] - correlation table of interaction radii r_jl(alpha).
    """
    # 1. Расчет array_alpha - направлений к центрам нагнетательных
    array_alpha = calculate_alpha(point_well, (centers_x, centers_y))

    # 2. Расчет array_lambda_k - проекций долей влияния нагнетательных на лучи с постоянной дельтой тетта
    list_angles = np.arange(0, 2 * np.pi + delta_theta, delta_theta)
    alpha_array = np.array(array_alpha)

    # Условия и коррекция углов
    mask1 = (list_angles[:, None] >= 3 / 2 * np.pi) & ((alpha_array >= 0) & (alpha_array <= np.pi / 2))
    alpha_corrected = np.where(mask1, alpha_array + 2 * np.pi, alpha_array)

    # Проекции
    mask2 = np.abs(alpha_corrected - list_angles[:, None]) <= np.pi / 2
    cos_diff = np.cos(alpha_corrected - list_angles[:, None])
    lambda_k = np.sum(interferense_values * cos_diff * mask2, axis=1)

    # 3. Масштабирование лучей по площади многоугольника
    eta = calculate_eta(r_eff, lambda_k, delta_theta)
    L_k_list = lambda_k * eta

    # 4. Расчет r_jl_values - таблицы радиусов взаимодействия от угла
    alpha_angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    r_jl_values = calculate_r_jl(L_k_list, delta_theta, alpha_angles)
    return alpha_angles, r_jl_values


def calculate_eta(R_x, lambdas, delta_theta):
    """
    Calculation of a scalable coefficient η = Rₓ * √(2π / (sin(Δθ) * Σ(λₖ * λₖ₊₁))).
    for vectors lambda_k to achieve the target polygon area size
    Args:
        R_x (int): effective radius of well [m]
        lambdas (np.array): rays lambda_k
        delta_theta (int) = np.pi / 10: Angle for calculating the projections of alpha_ij onto the rays lambda_k [rad]

    Returns:
        eta (int): scalable coefficient
    """
    S = calculate_S_polygon(lambdas, delta_theta)
    eta = R_x * math.sqrt(math.pi / (S + 1e-12))
    return eta


def calculate_S_polygon(lambdas, delta_theta):
    """
    Calculating the area of a polygon consisting of rays lambda_k.
    Args:
        lambdas (np.array): rays lambda_k
        delta_theta (int) = np.pi / 10: Angle for calculating the projections of alpha_ij onto the rays lambda_k [rad]

    Returns:
        S (int): area of a polygon
    """
    # Создаем пары соседних элементов: (λ₁, λ₂), (λ₂, λ₃), ..., (λ_M, λ_{M+1})
    pairs = zip(lambdas[:-1], lambdas[1:])
    # Вычисляем сумму произведений пар
    total_sum = sum(a * b for a, b in pairs)
    # Итоговый результат
    S = (math.sin(delta_theta) / 2) * total_sum
    return S


def calculate_r_jl(L_k_list, delta_theta, alpha):
    """
    Compute r_jl(alpha) with account of the circular structure of rays (0..2π).
    Vectorized version.
    Args:
        L_k_list (np.ndarray): Array of ray lengths L_k
        delta_theta (float): Angle step between ray [radians]
        alpha (np.ndarray or float): Angle(s) [radians]

    Returns:
        np.ndarray: r_jl values for each angle on alpha
    """
    # Преобразуем вход в numpy array, если это скаляр
    alpha = np.asarray(alpha)
    scalar_input = False
    if alpha.ndim == 0:
        alpha = alpha[None]  # Делаем 1D массив
        scalar_input = True

    num_rays = len(L_k_list)
    alpha_norm = alpha % (2 * np.pi)  # Нормализация угла к [0, 2π)
    # Вычисляем индексы k для каждого угла
    k = (alpha_norm // delta_theta).astype(int) % num_rays

    # Получаем соответствующие L_k и L_k+1
    L_k = L_k_list[k]
    L_k_plus_1 = L_k_list[(k + 1) % num_rays]

    # Вычисляем числитель и знаменатель
    numerator = L_k * L_k_plus_1 * np.sin(delta_theta)
    term1 = L_k_plus_1 * np.sin((k + 1) * delta_theta - alpha_norm)
    term2 = L_k * np.sin(alpha_norm - k * delta_theta)
    denominator = term1 + term2

    # Избегаем деления на ноль (можно добавить малую константу)
    epsilon = 1e-10
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)

    r_jl = numerator / denominator
    # Возвращаем скаляр, если на входе был скаляр
    return r_jl[0] if scalar_input else r_jl


def get_r_jl(target_angles, angles, r_jl_values):
    """
    Get r_jl for target angles based on correlation table r_jl(alpha).
    Args:
        target_angles(np.ndarray): Angle(s) [radians]
        angles (np.ndarray): angles of correlation table r_jl(alpha)
        r_jl_values (np.ndarray): r_jl of correlation table r_jl(alpha)

    Returns:
        np.ndarray: r_jl values for each angle on target_angles
    """
    # Находим индексы ближайших углов (бинарный поиск)
    idx = np.searchsorted(angles, target_angles, side="left")

    # Обрабатываем граничные случаи (когда target_angle выходит за пределы angles)
    idx = np.clip(idx, 1, len(angles) - 1)

    # Получаем соседние углы и соответствующие r_jl значения
    alpha_prev = angles[idx - 1]
    alpha_next = angles[idx]
    r_prev = r_jl_values[idx - 1]
    r_next = r_jl_values[idx]

    # Вычисляем веса для интерполяции
    # Если alpha_next == alpha_prev, то интерполяция невозможна, и мы берем r_prev
    mask = (alpha_next != alpha_prev)
    weights = np.zeros_like(target_angles)
    weights[mask] = (target_angles[mask] - alpha_prev[mask]) / (alpha_next[mask] - alpha_prev[mask])

    # Линейная интерполяция
    r_interpolated = r_prev + (r_next - r_prev) * weights
    return r_interpolated






