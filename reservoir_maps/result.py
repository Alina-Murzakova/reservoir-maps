import numpy as np
import logging
import warnings
from dataclasses import dataclass
from typing import Optional
from .input import (MapParams, ReservoirParams, FluidParams, RelativePermeabilityParams, Options, MapCollection,
                    validate_and_prepare_data_wells, dataclass_from_dict)
from .current_saturation import calculate_current_saturation
from .reserves import (calculate_oil_initially_in_place, calculate_initial_recoverable_reserves,
                       calculate_residual_recoverable_reserves)
from .water_cut import calculate_water_cut

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ResultMaps:
    """
    Container for the resulting maps.

    Attributes:
        data_So_current (np.ndarray): Current oil saturation map in dimensionless (H, W)
        data_water_cut (np.ndarray): Current water cut map in persent (H, W)
        data_OIIP (np.ndarray): Oil initially in place (OIIP)  map in t/ha (H, W)
        data_IRR (np.ndarray): Initial recoverable oil reserves map in t/ha (H, W)
        data_RRR (np.ndarray): Residual recoverable oil reserves map in t/ha (H, W)
    """
    data_So_current: np.ndarray  # current oil saturation
    data_water_cut: np.ndarray  # current water cut
    data_OIIP: np.ndarray  # oil initially in place
    data_IRR: np.ndarray  # initial recoverable oil reserves
    data_RRR: np.ndarray  # residual recoverable oil reserves


def get_maps(dict_maps: dict,
             dict_data_wells: dict,
             dict_map_params: dict,
             dict_reservoir_params: dict,
             dict_fluid_params: dict,
             dict_relative_permeability: dict,
             dict_options: Optional[dict] = None) -> ResultMaps:
    """
    Main function for getting all maps: saturation maps, water cut and recoverable reserves.
    Args:
        dict_maps: dictionary with input maps
        dict_data_wells: dictionary with well data
        dict_map_params: dictionary with the spatial resolution and mapping configuration
        dict_reservoir_params: dictionary with general properties of the reservoir
        dict_fluid_params: dictionary with parameters of reservoir fluids (oil, water)
        dict_relative_permeability: dictionary with parameters for relative phase permeability
        dict_options: optional calculation settings

    Returns:
        ResultMaps: an object containing the resulting maps
    """
    logger.info("Converting dict_data_wells to dataframe and preparing <data_wells>")
    data_wells = validate_and_prepare_data_wells(dict_data_wells)

    logger.info("Initializing input maps and params")
    logger.debug("Initializing maps")
    maps = dataclass_from_dict(MapCollection, dict_maps)

    logger.debug("Initializing map_params")
    map_params = dataclass_from_dict(MapParams, dict_map_params)

    logger.debug("Initializing reservoir_params")
    reservoir_params = dataclass_from_dict(ReservoirParams, dict_reservoir_params)

    logger.debug("Initializing fluid_params")
    fluid_params = dataclass_from_dict(FluidParams, dict_fluid_params)

    logger.debug("Initializing relative_permeability")
    relative_permeability = dataclass_from_dict(RelativePermeabilityParams, dict_relative_permeability)

    logger.debug("Initializing  options")
    options = dataclass_from_dict(Options, dict_options) if dict_options else Options()

    logger.info("Calculating current oil saturation <data_So_current>")
    data_So_current = calculate_current_saturation(maps, data_wells, map_params, reservoir_params, fluid_params,
                                                   relative_permeability, options)
    logger.info("Calculating current water cut <data_water_cut>")
    data_water_cut = calculate_water_cut(maps, data_So_current, fluid_params, relative_permeability)
    logger.debug("Calculating oil initially in place <data_OIIP>")
    data_OIIP, sum_OIIP = calculate_oil_initially_in_place(maps, map_params, fluid_params)
    logger.debug("Calculating initial recoverable reserves <data_IRR>")
    data_IRR, sum_IRR = calculate_initial_recoverable_reserves(data_OIIP, map_params, reservoir_params)
    logger.debug("Calculating residual recoverable reserves <data_RRR>")
    data_RRR, sum_RRR = calculate_residual_recoverable_reserves(maps, data_So_current, data_OIIP, map_params,
                                                                reservoir_params, fluid_params)
    relative_error_reserves = (((sum_IRR - sum_RRR) - data_wells.Qo_cumsum.sum()) / data_wells.Qo_cumsum.sum() * 100)
    logger.info(f"Relative error of reserves and production: {relative_error_reserves:.3f}%")
    if abs(relative_error_reserves) > 1.0:
        warnings.simplefilter("always", UserWarning)
        warnings.warn("Relative error of reserves and production exceeds tolerable error (1%), check: \n"
                      "- relative phase permeability \n"
                      "- current water cut of wells \n"
                      "- map of initial oil saturation", UserWarning, stacklevel=2)

    return ResultMaps(
        data_So_current,
        data_water_cut,
        data_OIIP,
        data_IRR,
        data_RRR
    )
