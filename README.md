# Reservoir-maps

**Reservoir-maps** is a Python package for calculating 2D reservoir state maps used in oil field development analysis.

### ✨ Main Features
* Current oil saturation map
* Water cut map
* Residual recoverable oil reserves (RRR) map

### ➕ Additional Features
* Oil initially in place (OIIP) map  
* Initial recoverable oil reserves (IRR) map

### 🔢 Workflow Stages  
1. Input Preparation
2. Saturation Calculation at Well Points (One-Phase Model)
3. Well Data Expansion to Trajectory Points
4. Influence Weights Calculation
5. Well Interference Calculation
6. Optimization & Interpolation: Material Balance Enforcement
7. Result Map Generation

## ⚙️ Installation
### Dependencies
`reservoir-maps` requires:
* Python (>= 3.8)
* NumPy (~=1.26.4)
* SciPy (~=1.14.0)
* Pandas (~=2.2.3)
* Scikit-image (~=0.25.2)

### User installation
To install from github:
```bash
pip install git+https://github.com/Alina-Murzakova/reservoir-maps.git
```

## 🚀 Usage

### 📥 Input Parameters

* `dict_maps`: dict
  
A dictionary of required reservoir maps.  

| Key                      | Type       | Description                | Unit |
|--------------------------|------------|----------------------------|:----:|
| `NNT`                    | np.ndarray | Net oil thickness map      |  m   |
| `initial_oil_saturation` | np.ndarray | Initial oil saturation map |  –   |
| `porosity`               | np.ndarray | Porosity map               |  –   |

* `dict_data_wells`: dict 

A dictionary of wells data arrays.

| Key            | Type         | Description                             |   Unit    |
|----------------|--------------|-----------------------------------------|:---------:|
| `well_number`  | str / int    | Well identifier                         |     —     |
| `work_marker`  | str          | Marker 'prod' or 'inj' well             |     —     |
| `no_work_time` | int / float  | Time since well was inactive            |  months   |
| `Qo_cumsum`    | int / float  | Cumulative oil production               |    ton    |
| `Winj_cumsum`  | int / float  | Cumulative water injection              |    m³     |
| `water_cut`    | int / float  | The latest water cut                    | fraction  |
| `r_eff`        | int / float  | Effective drainage radius               |     m     |
| `NNT`          | int / float  | Net oil thickness                       |     m     |
| `permeability` | int / float  | Reservoir permeability at well location |    mD     |
| `T1_x_pix`     | int / float  | X coordinate of T1 point in pixels      |    pix    |
| `T1_y_pix`     | int / float  | Y coordinate of T1 point in pixels      |    pix    |
| `T3_x_pix`     | int / float  | X coordinate of T3 point in pixels      |    pix    |
| `T3_y_pix`     | int / float  | Y coordinate of T3 point in pixels      |    pix    |

* `dict_map_params`: dict   

A dictionary of input parameters that define the map configuration.

| Key               | Type    | Description                       |    Unit    |
|-------------------|---------|-----------------------------------|:----------:|
| `size_pixel`      | int     | Size of one pixel in the map grid |     m      |
| `switch_fracture` | boolean | Enable fracture modeling          | True/False |


* `dict_reservoir_params`: dict  

A dictionary of general properties of the reservoir.

| Key                   | Type  | Description                              | Unit |
|-----------------------|-------|------------------------------------------|:----:|
| `KIN`                 | float | Recovery factor [0; 1]                   |  –   |
| `azimuth_sigma_h_min` | float | Azimuth of the minimum horizontal stress | deg  |
| `l_half_fracture`     | float | Half-length of hydraulic fracture        |  m   |


* `dict_fluid_params`: dict   

A dictionary of reservoir fluids (oil, water).

| Key        | Type  | Description                   | Unit  |
|------------|-------|-------------------------------|:-----:|
| `pho_surf` | float | Surface oil density           | g/cm³ |
| `mu_o`     | float | Oil viscosity                 |  cP   |
| `mu_w`     | float | Water viscosity               |  cP   |
| `Bo`       | float | Oil formation volume factor   | m³/m³ |
| `Bw`       | float | Water formation volume factor | m³/m³ |

* `dict_relative_permeability`: dict

A dictionary of relative phase permeability.

| Key   | Type  | Description                              |
|-------|-------|------------------------------------------|
| `Swc` | float | Connate water saturation                 |
| `Sor` | float | Residual oil saturation                  |
| `Fw`  | float | End-point relative permeability of water |
| `m1`  | float | Corey exponent for water phase           |
| `Fo`  | float | End-point relative permeability of oil   |
| `m2`  | float | Corey exponent for oil phase             |

* `dict_options`: dict, optional  

A dictionary of additional calculation options.

| Key     | Type  | Description                          | Default  |
|---------|-------|--------------------------------------|:--------:|
| `betta` | float | Power coefficient for well influence |   1.5    |
| `delta` | float | Decay rate of well influence         |  0.0001  |

### 💡 Example
Here’s a minimal example of how to use `reservoir_maps`:
```python
import numpy as np
from reservoir_maps import get_maps

# Prepare your input dictionaries
dict_maps = {"NNT": np.ones((10, 10)), "initial_oil_saturation": np.ones((10, 10)), "porosity": np.ones((10, 10))}
dict_data_wells = {'well_number': [1, 2],
                   'work_marker': ['prod', 'inj'],
                   'no_work_time': [0.0, 10.0],
                   'Qo_cumsum': [1000, 0],
                   'Winj_cumsum': [0, 2000],
                   'water_cut': [10, 0.0],
                   'r_eff': [100, 120],
                   'NNT': [5.0, 6],
                   'permeability': [20, 15.0],
                   'T1_x_pix': [1, 6],
                   'T1_y_pix': [1, 6],
                   'T3_x_pix': [5, 9],
                   'T3_y_pix': [2, 6],
                   }
dict_map_params = {"size_pixel": 50, "switch_fracture": False}
dict_reservoir_params = {"KIN": 0.25, "azimuth_sigma_h_min": 30, "l_half_fracture": 100}
dict_fluid_params = {"pho_surf": 0.850, "mu_o": 0.75, "mu_w": 0.3, "Bo": 1.2, "Bw": 1.0}
dict_relative_permeability = {"Sor": 0.3, "Swc": 0.2, "Fw": 0.3, "m1": 1, "Fo": 1, "m2": 1.0}

# Calculate maps
result = get_maps(dict_maps,
                  dict_data_wells,
                  dict_map_params,
                  dict_reservoir_params,
                  dict_fluid_params,
                  dict_relative_permeability,
                  )

# Access results
map_So_current = result.data_So_current  # current oil saturation array
map_water_cut = result.data_water_cut  # water cut array
map_OIIP = result.data_OIIP  # oil initially in place (OIIP) array
map_IRR= result.data_IRR  # initial recoverable oil reserves (IRR) array
map_RRR = result.data_RRR  # residual recoverable reserves (RRR) array
```

👉 For full examples, check: 
* [Jupyter Notebook](examples/reservoir_state_maps.ipynb) 
* [Python run example](examples/run_example.py)

## 👷 Who should use Reservoir-maps?
**Reservoir-maps** is an open-source package for:  
* Reservoir engineers 
* Oil & gas researchers 
* Field development analysts 
* Students and educators in petroleum engineering

## 📄 License
This project is licensed under the [MIT License](https://github.com/Alina-Murzakova/reservoir-maps/blob/main/LICENSE).

## 🙋‍♀️ Authors
* [Alina Murzakova](https://github.com/Alina-Murzakova)
* [Anastasia Rybakovskaya](https://github.com/ryba-kovskaya)

## 🧪 Tests
Basic test cases are located in the [tests](tests). To run them:
```bash
pytest
```
Make sure you have pytest installed: pip install pytest.

## 📬 Feedback & Contributions
We welcome feedback, issues and pull requests!