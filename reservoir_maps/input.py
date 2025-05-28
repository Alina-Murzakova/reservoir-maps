import numpy as np
import pandas as pd
import logging

from dataclasses import dataclass, fields, MISSING
from typing import Optional, Type, TypeVar
from .utils import get_line_cells

T = TypeVar("T")
logger = logging.getLogger(__name__)


def validate_and_prepare_data_wells(dict_data_wells: dict) -> pd.DataFrame:
    """

    :param dict_data_wells:
    :return:
    data_wells -> pd.DataFrame
    """
    if not isinstance(dict_data_wells, dict):
        raise TypeError("Expected dict_data_wells in parameters")

    # Проверка наличия обязательных полей
    REQUIRED_KEYS = ["well_number", "work_marker", "no_work_time", "Qo_cumsum", "Winj_cumsum",
                     "water_cut", "r_eff", "NNT", "permeability",
                     "T1_x_pix", "T1_y_pix", "T3_x_pix", "T3_y_pix",
                     ]
    VALID_WORK_MARKER = {"inj", "prod"}

    missing = [k for k in REQUIRED_KEYS if k not in dict_data_wells]
    if missing:
        raise ValueError(f"Missing required keys in dict_data_wells: {missing}")

    # Преобразование в np.array
    arrays_data_wells = {k: np.asarray(dict_data_wells[k]) for k in REQUIRED_KEYS}
    # Проверка длины массивов
    lengths = {k: len(v) for k, v in arrays_data_wells.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All input arrays must be the same length: {lengths}")

    # Проверка допустимых значений type_well
    unique_work_marker = set(arrays_data_wells["work_marker"])
    if not unique_work_marker.issubset(VALID_WORK_MARKER):
        raise ValueError(f"Invalid work_marker: {unique_work_marker}. Allowed: {VALID_WORK_MARKER}")

    # Преобразование в DataFrame
    data_wells = pd.DataFrame(arrays_data_wells)
    # Фильтрация
    data_wells = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    # Создание списков координат сетки, принадлежащих скважинам
    logger.info("Creating points of wells's trajectory")
    data_wells['trajectory_x'], data_wells['trajectory_y'] = zip(*data_wells.apply(
        lambda row: get_line_cells(int(row['T1_x_pix']), int(row['T1_y_pix']), int(row['T3_x_pix']),
                                   int(row['T3_y_pix'])), axis=1))
    return data_wells


def validate_and_prepare_maps(dict_maps: dict) -> dict:
    if not isinstance(dict_maps, dict):
        raise TypeError("Expected dict_maps in parameters")

    REQUIRED_KEYS = ['NNT', 'initial_oil_saturation', 'porosity']

    missing = [k for k in REQUIRED_KEYS if k not in dict_maps]
    if missing:
        raise ValueError(f"Missing required maps in dict_maps: {missing}")

    # Проверка типов
    for key, value in dict_maps.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Map '{key}' must be a numpy array, got {type(value)}")

    shapes = [v.shape for v in dict_maps.values()]
    if len(set(shapes)) > 1:
        raise ValueError(f"All map arrays must have the same shape. Got shapes: {shapes}")

    return dict_maps


@dataclass
class MapCollection:
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
            if isinstance(value, np.ndarray):
                if not np.issubdtype(value.dtype, np.number):
                    raise TypeError(f"'{name}' должен быть числовым массивом")
                if np.isnan(value).any():
                    raise ValueError(f"'{name}' содержит NaN")


@dataclass
class MapParams:
    size_pixel: int
    switch_fracture: bool = False

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class ReservoirParams:
    KIN: float
    azimuth_sigma_h_min: Optional[float] = None
    l_half_fracture: Optional[float] = None

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class FluidParams:
    pho_surf: float
    mu_o: float
    mu_w: float
    Bo: float
    Bw: float

    def __post_init__(self):
        validate_numbers(self)


@dataclass
class RelativePermeabilityParams:
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
    betta: float = 1.5
    delta: float = 0.0001

    def __post_init__(self):
        validate_numbers(self)


def validate_numbers(obj):
    for name, value in vars(obj).items():
        if value is None:
            raise ValueError(f"'{name}' не должно быть None")

        # Проверка на булев тип
        if name == 'switch_fracture':
            if not isinstance(value, bool):
                raise TypeError(f"'{name}' должно быть True или False (тип bool)")
            continue

        # Проверка на числовой тип
        if not isinstance(value, (int, float)):
            raise TypeError(f"'{name}' должно быть числом (int или float)")

        # Дополнительные ограничения по диапазонам
        if name in {"Swc", "Sor", "Fw", "Fo", "KIN"}:
            if not (0 <= value <= 1):
                raise ValueError(f"'{name}' должно быть в диапазоне [0, 1]")
        elif value <= 0:
            raise ValueError(f"'{name}' должно быть положительным числом")


def dataclass_from_dict(cls: Type[T], data: Optional[dict]) -> T:
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
