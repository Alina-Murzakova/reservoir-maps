import math
from skimage.draw import line


def correction_injection_well_trajectory(coords_x, coords_y, sigma_h, l_half_fracture):
    """Переориентация траектории нагнетательных скважин, учитывая направление минимального горизонтального напряжения"""
    # Определеим координаты центра трещины
    center_x, center_y = coords_x[len(coords_x) // 2], coords_y[len(coords_y) // 2]
    # Азимут трещины
    azimuth_fracture = sigma_h + math.pi / 2
    new_t1_x, new_t1_y = center_x + l_half_fracture * math.cos(azimuth_fracture), center_y + l_half_fracture * math.sin(
        azimuth_fracture)
    new_t3_x, new_t3_y = center_x - l_half_fracture * math.cos(azimuth_fracture), center_y - l_half_fracture * math.sin(
        azimuth_fracture)
    return get_line_cells(new_t1_x, new_t1_y, new_t3_x, new_t3_y)


def get_line_cells(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return [x1], [y1]
    rr, cc = line(int(y1), int(x1), int(y2), int(x2))  # skimage.draw.line возвращает индексы
    return cc.tolist(), rr.tolist()