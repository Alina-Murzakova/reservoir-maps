import pickle
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from calculation_parameters import constants
from reservoir_maps.result import get_maps

import logging
logging.basicConfig(
    level=logging.INFO,
)


# Loading dataframe with well data
with open(f"data_wells.pkl", 'rb') as f:
    data_wells = pickle.load(f)

# Loading metadata
with open("maps_metadata.json", "r", encoding="utf-8") as f:
    maps_metadata = json.load(f)


keys_data_wells = list(data_wells.columns)
# Preparing a dictionary with well data
dict_data_wells = {key: np.asarray(data_wells[key]) for key in keys_data_wells}

# Preparing a dictionary with maps
dict_maps = {}
for meta in maps_metadata:
    data = np.load(os.path.join(meta["data_file"]))
    type_map = meta["type_map"]
    dict_maps[meta["type_map"]] = data

# Constants and calculation options
map_parameters = constants['map_params']
reservoir_params = constants['reservoir_params']
fluid_params = constants['fluid_params']
relative_permeability = constants['relative_permeability']

# Result
res = get_maps(dict_maps, dict_data_wells, map_parameters, reservoir_params, fluid_params, relative_permeability)

maps = {
    "Current oil saturation": res.data_So_current,
    "Water cut": res.data_water_cut,
    "Oil initially in place (OIIP)": res.data_OIIP,
    "Initial recoverable oil reserves (IRR)": res.data_IRR,
    "Residual recoverable oil reserves (RRR)": res.data_RRR
}

for name, data in maps.items():
    data = np.where(data == 1.70141E+0038, 0.0, data)
    plt.figure()
    plt.imshow(data, origin="upper")
    plt.colorbar()
    plt.title(name)
    plt.savefig(f"{name}.png", dpi=500)
    plt.close()
