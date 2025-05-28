# Reservoir-maps

**Reservoir-maps** is a Python package for calculating 2D reservoir state maps used in oil field development analysis.  

### ✨ Main Features
* Current oil saturation map
* Water cut map
* Residual recoverable oil reserves (RRR) map


### ➕ Additional Features
* Oil initially in place (OIIP) map  
* Initial recoverable oil reserves (IRR) map

## Installation
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
pip install git+https://github.com/Alina-Murzakova/reservoir_maps.git
```

## Usage
Here’s a minimal example of how to use `reservoir_maps`:
```python
from reservoir_maps import get_maps

# Prepare your input dictionaries
dict_maps = {...}
dict_data_wells = {...}
dict_map_params = {...}
dict_reservoir_params = {...}
dict_fluid_params = {...}
dict_relative_permeability = {...}
dict_options = {...}

# Calculate maps
result = get_maps(
    dict_maps,
    dict_data_wells,
    dict_map_params,
    dict_reservoir_params,
    dict_fluid_params,
    dict_relative_permeability,
    dict_options
)

# Access results
result.data_So_current  # current oil saturation array
result.data_water_cut   # water cut array
result.data_OIIP   # oil initially in place (OIIP) array
result.data_IRR   # initial recoverable oil reserves (IRR) array
result.data_RRR   # residual recoverable reserves (RRR) array
```
For full examples, see the examples notebook (Jupyter Notebook).

## 👷 Who should use Reservoir-maps?
**Reservoir-maps** is an open-source package for:  
* Reservoir engineers 
* Oil & gas researchers 
* Field development analysts 
* Students and educators in petroleum engineering

## 📄 License
This project is licensed under the  [MIT License]().

## 🙋‍♀️ Authors
* Alina Murzakova
* Anastasia Rybakovskaya