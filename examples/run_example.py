import pickle
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXAMPLE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calculation_parameters import constants
from reservoir_maps.result import get_maps

import logging
logging.basicConfig(
    level=logging.INFO,
)


# Loading dataframe with well data
with (EXAMPLE_DIR / "data_wells.pkl").open('rb') as f:
    data_wells = pickle.load(f)

# Loading metadata
with (EXAMPLE_DIR / "maps_metadata.json").open("r", encoding="utf-8") as f:
    maps_metadata = json.load(f)


keys_data_wells = list(data_wells.columns)
# Preparing a dictionary with well data
dict_data_wells = {key: np.asarray(data_wells[key]) for key in keys_data_wells}

# Preparing a dictionary with maps
dict_maps = {}
for meta in maps_metadata:
    data = np.load(EXAMPLE_DIR / meta["data_file"])
    type_map = meta["type_map"]
    dict_maps[meta["type_map"]] = data

# Constants and calculation options
map_parameters = constants['map_params']
reservoir_params = constants['reservoir_params']
fluid_params = constants['fluid_params']
relative_permeability = constants['relative_permeability']

# Result
start_time = time.time()
res = get_maps(dict_maps, dict_data_wells, map_parameters, reservoir_params, fluid_params, relative_permeability)
elapsed = time.time() - start_time
print(f"Time: {elapsed / 60:.1f}min")

maps = {
    "Current oil saturation": res.data_So_current,
    "Water cut": res.data_water_cut,
    "Oil initially in place (OIIP)": res.data_OIIP,
    "Initial recoverable oil reserves (IRR)": res.data_IRR,
    "Residual recoverable oil reserves (RRR)": res.data_RRR
}

print(f"Relative error of reserves and production: {res.rel_error_RRR:.3f}%")
print("Adapted relative permeability by well:")
print(res.adapted_relative_permeability)

for name, data in maps.items():
    data = np.where(data == 1.70141E+0038, 0.0, data)
    plt.figure()
    plt.imshow(data, origin="upper")
    plt.colorbar()
    plt.title(name)
    plt.savefig(EXAMPLE_DIR / f"{name}.png", dpi=500)
    plt.close()
