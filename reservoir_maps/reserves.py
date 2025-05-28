import numpy as np
from .input import MapCollection, MapParams, FluidParams


def calculate_oil_initially_in_place(maps: MapCollection,
                                     map_params: MapParams,
                                     fluid_params: FluidParams):
    """
    Calculates oil initially in place (OIIP) across the grid (map).
    Args:
        maps: Collection of input maps
        map_params: Parameters for accounting in maps
        fluid_params: Parameters of the fluids (oil, water)
    Returns:
        2D array of OIIP -  oil initially in place (map)
    """
    # умножение на 10000 требуется для перевода единиц измерения плотности запасов из тонн/м2 в тонн/га
    data_OIIP = maps.NNT * maps.initial_oil_saturation * maps.porosity * fluid_params.pho_surf * 10000 / fluid_params.Bo
    sum_OIIP = np.sum(data_OIIP * map_params.size_pixel ** 2 / 10000)  # обратно перевод га в м2
    print(f"НГЗ NGT - 84882 тыс.т")
    print(f"НГЗ расчет через ННТ - {sum_OIIP / 1000} тыс.т")
    return data_OIIP, sum_OIIP


def calculate_initial_recoverable_reserves(data_OIIP,
                                           map_params,
                                           reservoir_params):
    """
    Calculates initial recoverable oil reserves (IRR) across the grid (map).
    Args:
        data_OIIP:  2D array of OIIP -  oil initially in place
        map_params: Parameters for accounting in maps
        reservoir_params: General properties of the reservoir
    Returns:
        2D array of IRR -  initial recoverable oil reserves (map)
    """
    # НИЗ - initial recoverable oil reserves
    print('НИЗ NGT - 28457.61 тыс.т')
    data_IRR = data_OIIP * reservoir_params.KIN
    sum_IRR = np.sum(data_IRR * map_params.size_pixel ** 2 / 10000)  # обратно перевод га в м2
    print(f'НИЗ расчет - {sum_IRR / 1000} тыс.т')
    return data_IRR, sum_IRR


def calculate_residual_recoverable_reserves(maps, data_So_current, data_OIIP, map_params, reservoir_params, fluid_params):
    # ОГЗ - remaining oil in place
    data_ROIP = maps.NNT * data_So_current * maps.porosity * fluid_params.pho_surf * 10000 / fluid_params.Bo
    sum_ROIP = np.sum(data_ROIP * map_params.size_pixel ** 2 / 10000)
    print('ОГЗ NGT - 76005.04 тыс.т')
    print(f'ОГЗ расчет - {sum_ROIP / 1000} тыс.т')
    # ОИЗ - residual recoverable reserves
    data_RRR = data_ROIP - data_OIIP * (1 - reservoir_params.KIN)
    data_RRR[data_RRR < 0] = 0
    sum_RRR = np.sum(data_RRR * map_params.size_pixel ** 2 / 10000)
    print('ОИЗ NGT - 20131.86 тыс.т')
    print(f'ОИЗ расчет - {sum_RRR / 1000} тыс.т')
    return data_RRR, sum_RRR
