import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional
from .input import (MapParams, ReservoirParams, FluidParams, RelativePermeabilityParams, Options, MapCollection,
                    validate_and_prepare_data_wells, dataclass_from_dict)
from .current_saturation import calculate_current_saturation
from .reserves import (calculate_oil_initially_in_place, calculate_initial_recoverable_reserves,
                       calculate_residual_recoverable_reserves)
from .water_cut import calculate_water_cut

logger = logging.getLogger(__name__)


@dataclass
class ResultMaps:
    data_So_current: np.ndarray  # current oil saturation
    data_water_cut: np.ndarray  # current water cut
    data_OIIP: np.ndarray  # oil initially in place
    data_IRR: np.ndarray  # initial recoverable reserves of oil
    data_RRR: np.ndarray  # residual recoverable reserves


def get_maps(dict_maps: dict,
             dict_data_wells: dict,
             dict_map_params: dict,
             dict_reservoir_params: dict,
             dict_fluid_params: dict,
             dict_relative_permeability: dict,
             dict_options: Optional[dict] = None) -> ResultMaps:
    """

    Args:
        dict_maps:
        dict_data_wells:
        dict_map_params:
        dict_reservoir_params:
        dict_fluid_params:
        dict_relative_permeability:
        dict_options:

    Returns:
        ResultMaps - Объект с рассчитанными картами
    """
    logger.info("Converting dict_data_wells to dataframe and preparing data_wells")
    data_wells = validate_and_prepare_data_wells(dict_data_wells)

    logger.info("Initializing maps and params")
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
    options = dataclass_from_dict(Options, dict_options)

    logger.info("Calculating <data_So_current>")
    data_So_current = calculate_current_saturation(maps, data_wells, map_params, reservoir_params, fluid_params,
                                                   relative_permeability, options)
    logger.info("Calculating <data_water_cut>")
    data_water_cut = calculate_water_cut(maps, data_So_current, fluid_params, relative_permeability)
    logger.info("Calculating <data_OIIP>")
    data_OIIP, sum_OIIP = calculate_oil_initially_in_place(maps, map_params, fluid_params)
    logger.info("Calculating <data_IRR>")
    data_IRR, sum_IRR = calculate_initial_recoverable_reserves(data_OIIP, map_params, reservoir_params)
    logger.info("Calculating <data_RRR>")
    data_RRR, sum_RRR = calculate_residual_recoverable_reserves(maps, data_So_current, data_OIIP, map_params,
                                                                reservoir_params, fluid_params)
    logger.info(
        f"Oil production difference, %: {((sum_IRR - sum_RRR) - data_wells.Qo_cumsum.sum()) / data_wells.Qo_cumsum.sum() * 100}")
    return ResultMaps(data_So_current, data_water_cut, data_OIIP, data_IRR, data_RRR)
