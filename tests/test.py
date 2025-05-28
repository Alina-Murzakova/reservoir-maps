import pickle
import json
import numpy as np
import os

from calculation_parameters import constants
from reservoir_maps.result import get_maps

# Фрейм с параметрами скважин
with open(f"data_wells.pkl", 'rb') as f:
    data_wells = pickle.load(f)

# Загрузка метаданных
with open("maps_metadata.json", "r", encoding="utf-8") as f:
    maps_metadata = json.load(f)

del data_wells['r_eff']
data_wells = data_wells.rename(columns={'r_eff_not_norm': 'r_eff'})
keys_data_wells = list(data_wells.columns)
# Подготовка словаря с данными скважин
dict_data_wells = {key: np.asarray(data_wells[key]) for key in keys_data_wells}

# Подготовка словаря с данными карт
dict_maps = {}
for meta in maps_metadata:
    data = np.load(os.path.join(meta["data_file"]))
    type_map = meta["type_map"]
    dict_maps[meta["type_map"]] = data

# Константы и параметры расчета
map_parameters = constants['map_params']
reservoir_params = constants['reservoir_params']
fluid_params = constants['fluid_params']
relative_permeability = constants['relative_permeability']

import logging
logging.basicConfig(
    level=logging.INFO,                # уровень детализации
)

res = get_maps(dict_maps, dict_data_wells, map_parameters, reservoir_params, fluid_params, relative_permeability)


import matplotlib.pyplot as plt
maps = {
    "data_So_current": res.data_So_current,
    "data_water_cut": res.data_water_cut,
    "data_OIIP": res.data_OIIP,
    "data_IRR": res.data_IRR,
    "data_RRR": res.data_RRR
}

for name, data in maps.items():
    plt.figure()
    plt.imshow(data, origin="upper")
    plt.colorbar()
    plt.title(name)
    plt.savefig(f"{name}.png", dpi=500)
    plt.close()  # закрыть фигуру, чтобы не держать много открытых