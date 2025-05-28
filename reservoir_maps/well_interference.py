import numpy as np
import math

"""
-----Учет радиусов влияния в распределении нефтенасыщенности-----
Также как и в случае матрицы расстояний distances, нам необходимо получить для каждой точки скважины массив значений r_jl, 
которые зависят от направлений векторов alpha_ij из точки скважины к каждой ячейке поля. 

Будем действовать по следующему алгоритму:
1. Сначала рассчитаем матрицу направлений alpha_ij для каждой ячейки поля
2. Для точки рассчитаем корреляционную таблицу r_jl(alpha_ij) через функцию calculate_r_jl_values
3. Матрицу alpha_ij прогоним через функцию get_r_jl - и получим матрицу радиусов
4. Добавим получившийся массив в общий и используем в функции расчета нефтенас-ти + обновим эффективные радиусы скважин
5. !NB Если встречается точка нагнетательной скважины, то для нее используется базовая матрица r_jl 
    из эффективных радиусов точки (или 1?)"""


def get_matrix_r_ij(valid_points, well_coord, x, y, work_markers, r_eff, h, Qo_cumsum, Winj_cumsum, size_pixel):
    # Расчет локальных матриц взаимодействия для точек
    matrix_r_ij = np.empty((valid_points[:, 1].shape[0], len(x)), dtype=np.float32)
    index = 0
    for index_point in range(len(x)):
        x_point, y_point, work_marker_point, r_eff_n = (x[index_point], y[index_point],
                                                        work_markers[index_point], r_eff[index_point])
        if work_marker_point == 'prod' or work_marker_point == 'inj':
            # Строка из матрицы влияния с созависимыми нагнетательными
            interference_array, mask_general = calc_interference_matrix(np.array([x_point, y_point]), work_marker_point,
                                                                        well_coord, h, Winj_cumsum, Qo_cumsum,
                                                                        work_markers)
            # Центры данных нагнетательных скважин
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
            print(0)
            matrix_r_ij[:, index] = np.full(valid_points[:, 1].shape[0], r_eff_n / size_pixel)
            index += 1
    return matrix_r_ij


def calc_interference_matrix(point_coord, work_marker_point, grid_point_wells, array_h, array_Winj, array_Qo,
                             array_work_marker, max_distance=1000):
    """Расчет коэффициентов участия и влияния"""
    array_distance = np.linalg.norm(grid_point_wells - point_coord, axis=1)
    mask_nearest_points = (array_distance <= max_distance) & (array_distance > 0)
    mask_marker = array_work_marker != work_marker_point
    mask_general = mask_nearest_points & mask_marker
    if work_marker_point == 'prod':
        "Расчет коэффициентов влияния для добывающих скважин"
        lambda_ij = array_h[mask_general] * array_Winj[mask_general] / array_distance[mask_general]
        lambda_ij = lambda_ij / lambda_ij.sum()
    else:
        "Расчет коэффициентов участия для нагнетательных скважин"
        lambda_ij = array_h[mask_general] * array_Qo[mask_general] / array_distance[mask_general]
        lambda_ij = 1 - lambda_ij / lambda_ij.sum()
    return lambda_ij, mask_general


def calculate_alpha(point_center, point_cell):
    """
    Векторизованный расчет угла (в радианах) от `point_center` до точек `point_cell`.
    Все входные данные должны быть массивами NumPy.
    :param point_center: tuple[float, float]
        Центральная точка (x, y).
    :param point_cell: tuple[np.ndarray, np.ndarray]
        Две матрицы координат: (x_coords, y_coords).
    :return:
    np.ndarray
        Матрица углов [0, 2π) той же формы, что и входные массивы.
    """
    x1, y1 = point_center
    x2, y2 = point_cell  # Распаковываем массивы координат

    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx)
    return np.where(angle < 0, angle + 2 * np.pi, angle)


def calculate_r_jl_values(point_well, r_eff, interferense_values, centers_x, centers_y,
                          delta_theta=np.pi / 10, num_points=365):
    """Векторизованная версия функции."""
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

    # 4. Расчет r_jl_values - таблицы радиусов взамодействия от угла
    alpha_angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    r_jl_values = calculate_r_jl(L_k_list, delta_theta, alpha_angles)
    return alpha_angles, r_jl_values


def calculate_eta(R_x, lambdas, delta_theta):
    """Вычисляет η = Rₓ * √(2π / (sin(Δθ) * Σ(λₖ * λₖ₊₁)))."""
    S = calculate_S_polygon(lambdas, delta_theta)
    eta = R_x * math.sqrt(math.pi / (S + 1e-12))
    return eta


def calculate_S_polygon(lambdas, delta_theta):
    # Создаем пары соседних элементов: (λ₁, λ₂), (λ₂, λ₃), ..., (λ_M, λ_{M+1})
    pairs = zip(lambdas[:-1], lambdas[1:])
    # Вычисляем сумму произведений пар
    total_sum = sum(a * b for a, b in pairs)
    # Итоговый результат
    S = (math.sin(delta_theta) / 2) * total_sum
    return S


def calculate_r_jl(L_k_list, delta_theta, alpha):
    """
    Вычисляет r_jl(alpha) с учётом кольцевой структуры лучей (0..2π).
    Векторизованная версия
    :param L_k_list: (np.ndarray) Массив длин лучей L_k.
    :param delta_theta: (float) Угловой шаг между лучами (в радианах).
    :param alpha: (np.ndarray или float) Угол(ы) (в радианах), может быть массивом.
    :return:
    np.ndarray: Значения r_jl для каждого угла в alpha.
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






