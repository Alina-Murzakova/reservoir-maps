import numpy as np
import logging

from .input import MapCollection, MapParams, FluidParams

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    logger.info(f"Calculated oil initially in place (OIIP): {sum_OIIP / 1000} thousand tons")
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
    data_IRR = data_OIIP * reservoir_params.KIN
    sum_IRR = np.sum(data_IRR * map_params.size_pixel ** 2 / 10000)  # обратно перевод га в м2
    logger.info(f"Calculated initial recoverable oil reserves (IRR): {sum_IRR / 1000} thousand tons")
    return data_IRR, sum_IRR


def calculate_residual_recoverable_reserves(maps, data_So_current, data_OIIP, map_params, reservoir_params, fluid_params):
    # ОГЗ - remaining oil in place
    data_ROIP = maps.NNT * data_So_current * maps.porosity * fluid_params.pho_surf * 10000 / fluid_params.Bo
    sum_ROIP = np.sum(data_ROIP * map_params.size_pixel ** 2 / 10000)
    logger.info(f"Calculated remaining oil in place (ROIP): {sum_ROIP / 1000} thousand tons")
    # ОИЗ - residual recoverable reserves
    data_RRR = data_ROIP - data_OIIP * (1 - reservoir_params.KIN)
    data_RRR[data_RRR < 0] = 0
    sum_RRR = np.sum(data_RRR * map_params.size_pixel ** 2 / 10000)
    logger.info(f"Calculated remaining oil in place (RRR): {sum_RRR / 1000} thousand tons")
    return data_RRR, sum_RRR
