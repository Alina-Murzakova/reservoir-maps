import numpy as np
import pandas as pd
import logging
import os

from dataclasses import dataclass, fields, MISSING
from typing import Optional, Type, TypeVar, Any
from .utils import get_line_cells
from pathlib import Path

T = TypeVar("T")
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def validate_and_prepare_data_wells(dict_data_wells: dict) -> pd.DataFrame:
    """
    Validates and preprocess data_wells from a dictionary.

    This function performs the following checks and transformations:
    - Verifies that the input is a dictionary.
    - Checks for the presence of all required fields.
    - Ensures all data arrays have the same length.
    - Validates that 'work_marker' values are either 'inj' or 'prod'.
    - Converts input arrays into a pandas DataFrame.
    - Filters out wells with zero cumulative production and injection.
    - Calculates well trajectories (grid cells along the T1-T3 line)
      and adds them as 'trajectory_x' and 'trajectory_y' columns.

    Args:
        dict_data_wells: A dictionary containing wells data arrays with the following keys:
            Required keys:
                - 'well_number' (str or int): well identifiers
                - 'work_marker' (str): 'prod' or 'inj'
                - 'no_work_time' (float): time since well was inactive [months]
                - 'Qo_cumsum' (float): cumulative oil production [t]
                - 'Winj_cumsum' (float): cumulative water injection [m³]
                - 'water_cut' (float): the latest water cut [%]
                - 'r_eff' (float): effective drainage radius [m]
                - 'NNT' (float): net oil thickness [m]
                - 'permeability' (float): reservoir permeability [mD]
                - 'T1_x_pix', 'T1_y_pix' (int): Start point of well trajectory (pixel coords).
                - 'T3_x_pix', 'T3_y_pix' (int): End point of well trajectory (pixel coords).

    Returns:
        pd.DataFrame: A validated and cleaned DataFrame data_wells with well attributes.

    Raises:
        TypeError: If input is not a dictionary or values have incorrect types.
        ValueError: If required keys are missing, arrays have mismatched lengths,
                    or invalid values are found.
    """
    if not isinstance(dict_data_wells, dict):
        raise TypeError("Expected dict_data_wells in parameters")

    REQUIRED_KEYS = ["well_number", "work_marker", "no_work_time", "Qo_cumsum", "Winj_cumsum",
                     "water_cut", "r_eff", "NNT", "permeability",
                     "T1_x_pix", "T1_y_pix", "T3_x_pix", "T3_y_pix",
                     ]
    EXPECTED_TYPES = {
        "well_number": (str, int),
        "work_marker": (str,),
        "no_work_time": (int, float),
        "Qo_cumsum": (int, float),
        "Winj_cumsum": (int, float),
        "water_cut": (int, float),
        "r_eff": (int, float),
        "NNT": (int, float),
        "permeability": (int, float),
        "T1_x_pix": (int, float),
        "T1_y_pix": (int, float),
        "T3_x_pix": (int, float),
        "T3_y_pix": (int, float),
    }
    VALID_WORK_MARKER = {"inj", "prod"}
    # Проверка наличия обязательных полей
    missing = [k for k in REQUIRED_KEYS if k not in dict_data_wells]
    if missing:
        raise ValueError(f"Missing required keys in dict_data_wells: {missing}")
    logger.debug("Missing keys in <dict_data_wells> not found.")

    for key, value in dict_data_wells.items():
        if value is None:
            raise ValueError(f"'{key}' must not be None")

    # Преобразование в np.array
    arrays_data_wells = {k: np.asarray(dict_data_wells[k]) for k in REQUIRED_KEYS}
    # Проверка длины массивов
    lengths = {k: len(v) for k, v in arrays_data_wells.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All input arrays must be the same length: {lengths}")
    logger.debug("All input arrays have the same length.")

    # Проверка допустимых значений type_well
    unique_work_marker = set(arrays_data_wells["work_marker"])
    if not unique_work_marker.issubset(VALID_WORK_MARKER):
        raise ValueError(f"Invalid work_marker: {unique_work_marker}. Allowed: {VALID_WORK_MARKER}")

    # Преобразование в DataFrame
    data_wells = pd.DataFrame(arrays_data_wells)
    # Проверка: нет пропущенных значений
    if data_wells.isna().any().any():
        nan_columns = data_wells.columns[data_wells.isna().any()].tolist()
        raise ValueError(f"Missing values (NaN or None) detected in columns: {nan_columns}")
    # Проверка типов значений
    for col, expected_types in EXPECTED_TYPES.items():
        invalid_mask = ~data_wells[col].dropna().map(lambda x: isinstance(x, expected_types))
        if invalid_mask.any():
            bad_values = data_wells.loc[invalid_mask.index[invalid_mask], col].unique()
            raise TypeError(
                f"Column '{col}' must contain only values of type {expected_types}, "
                f"but found values: {bad_values[:5]}"
            )
    logger.debug("Converting of the dictionary <dict_data_wells> to DataFrame <data_wells> completed successfully")
    # Фильтрация
    data_wells = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    # Создание списков координат сетки, принадлежащих скважинам
    data_wells['trajectory_x'], data_wells['trajectory_y'] = zip(*data_wells.apply(
        lambda row: get_line_cells(int(row['T1_x_pix']), int(row['T1_y_pix']), int(row['T3_x_pix']),
                                   int(row['T3_y_pix'])), axis=1))
    logger.debug("Creating points of wells's trajectory completed successfully")
    return data_wells


@dataclass
class MapCollection:
    """
    Container for input reservoir maps.

    Attributes:
        NNT (np.ndarray): Net oil thickness map in meters (H, W)
        initial_oil_saturation (np.ndarray): Initial oil saturation map in dimensionless (H, W)
        porosity (np.ndarray): Porosity map in dimensionless (H, W)
    """
    NNT: np.ndarray
    initial_oil_saturation: np.ndarray
    porosity: np.ndarray

    def __post_init__(self):
        self._validate_shapes()
        self._validate_numerical()

    def _validate_shapes(self):
        shapes = [v.shape for v in vars(self).values() if isinstance(v, np.ndarray)]
        if len(set(shapes)) > 1:
            raise ValueError(f"All map arrays must have the same shape. Got shapes: {shapes}")

    def _validate_numerical(self):
        for name, value in vars(self).items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Map '{name}' must be a numpy.ndarray")

            if value.size == 0:
                raise ValueError(f"Map '{name}' is empty")

            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"Map '{name}' must contain numeric values, got dtype={value.dtype}")

            if np.isnan(value).any():
                raise ValueError(f"Map '{name}' contains NaN")

            if np.isinf(value).any():
                raise ValueError(f"Map '{name}' contains Inf")


@dataclass
class MapParams:
    """
    Parameters related to the spatial resolution and mapping configuration.

    Attributes:
        size_pixel (int): Size of one pixel (cell) in the map grid [m]
        switch_fracture (bool): Enable fracture modeling [True/False]
        no_data_value (Optional[float]): Nodata value to ignore in maps (default: 1.70141E+0038)
    """
    size_pixel: int
    switch_fracture: bool = False
    no_data_value: Optional[float] = 1.70141E+0038

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class ReservoirParams:
    """
    Parameters describing general properties of the reservoir.

    Attributes:
        KIN (float): Recovery factor (dimensionless, between 0 and 1)
        azimuth_sigma_h_min (Optional[float]): Azimuth of minimum horizontal stress [degrees]
        l_half_fracture (Optional[float]): Half-length of hydraulic fracture [m]
    """
    KIN: float
    azimuth_sigma_h_min: Optional[float] = None
    l_half_fracture: Optional[float] = None

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class FluidParams:
    """
    Parameters of reservoir fluids (oil, water)

    Attributes:
        pho_surf (float): Surface oil density [g/cm³]
        mu_o (float): Oil viscosity [mPa·s]
        mu_w (float): Water viscosity [mPa·s]
        Bo (float): Oil formation volume factor [m³/m³]
        Bw (float): Water formation volume factor [m³/m³]
    """
    pho_surf: float
    mu_o: float
    mu_w: float
    Bo: float
    Bw: float

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class RelativePermeabilityParams:
    """
    Parameters for relative phase permeability.

    Attributes:
        Swc (float): Connate water saturation (dimensionless, between 0 and 1)
        Sor (float): Residual oil saturation (dimensionless, between 0 and 1)
        Fw (float): End-point relative permeability of water
        m1 (float): Corey exponent for water phase
        Fo (float): End-point relative permeability of oil
        m2 (float): Corey exponent for oil phase
    """
    Swc: float
    Sor: float
    Fw: float
    m1: float
    Fo: float
    m2: float

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class Options:
    """
    Additional calculation options

    Attributes:
        betta (float): Power coefficient for well interference influence (default: 2.0)
        delta (float): Coefficient controlling decay rate of well influence dependent on inactive time and permeability (default: 0.0001)
        max_distance (float): Maximum distance for the nearest surrounding (influencing) wells [m] (default: 1000)
        max_memory_gb (float):  Maximum allowed memory usage in gigabytes [GB] (default: 8.0)
        batch_size (int): Number of grid cells to process per batch (default: 50_000)
        tmp_dir (Optional[str]): Parent directory for temporary calculation files
    """
    betta: float = 2.0
    delta: float = 0.0001
    max_distance: float = 1000
    batch_size: int = 50_000
    tmp_dir: Optional[str] = None
    max_memory_gb: Optional[float] = None

    def __post_init__(self):
        self._validate_none_values()
        validate_numbers(self)

    def _validate_none_values(self):

        if self.tmp_dir is None:
            return

        if not isinstance(self.tmp_dir, str):
            raise TypeError("'tmp_dir' must be a string or None")

        path = Path(self.tmp_dir)

        # путь не должен указывать на файл
        if path.exists() and not path.is_dir():
            raise ValueError(f"'tmp_dir' must be a directory, got file: {path}")

        if self.max_memory_gb is None:
            return
        else:
            validate_numbers(self.max_memory_gb)


def validate_numbers(obj: Any) -> None:
    """
    Validates numeric and boolean attributes  of a dataclass-like object.
    """
    for name, value in vars(obj).items():
        # Skip tmp_dir
        if name == "tmp_dir":
            continue

        if name == "max_memory_gb" and value is None:
            continue

        if value is None:
            raise ValueError(f"'{name}' must not be None")

        # Check for boolean type
        if name == 'switch_fracture':
            if not isinstance(value, bool):
                raise TypeError(f"'{name}' must be True or False (type bool)")
            continue

        if name == 'no_data_value':
            if not isinstance(value, float):
                raise TypeError(f"'{name}' must be float")
            continue

        if name == 'batch_size':
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'{name}' must be a positive integer")
            continue

        # Check for numeric type
        if not isinstance(value, (int, float)):
            raise TypeError(f"'{name}' must be a number (int or float)")

        # Additional range limitations
        if name in {"Swc", "Sor", "Fw", "Fo", "KIN"}:
            if not (0 <= value <= 1):
                raise ValueError(f"'{name}' must be in the range [0, 1]")
        elif value <= 0:
            raise ValueError(f"'{name}' must be a positive number")


def dataclass_from_dict(cls: Type[T], data: Optional[dict]) -> T:
    """
    Creates a dataclass instance from a dictionary, using default values where applicable.

    This function initializes a dataclass from the provided dictionary `data`.
    If any required fields are missing and don't have a default value, it raises an error.

    Args:
        cls: The dataclass type to instantiate
        data: A dictionary of values to use for instantiation

    Returns:
         T: An instance of the dataclass `cls`.
    """
    # Использует значения по умолчанию из dataclass
    if data is None:
        data = {}
    missing = []
    init_values = {}
    for field in fields(cls):
        if field.name in data:
            init_values[field.name] = data[field.name]
        elif field.default is not MISSING or field.default_factory is not MISSING:
            continue
        else:
            # обязательное поле без значения
            missing.append(field.name)
    if missing:
        raise ValueError(f"Missing required data in {cls.__name__}: {missing}")
    return cls(**init_values)
