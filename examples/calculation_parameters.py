constants = {'map_params': {'size_pixel': 50,  # Size of one pixel (cell) in the map grid [m]
                            'switch_fracture': False,  # Enable fracture modeling [True/False]
                            },

             'reservoir_params': {'KIN': 0.336,  # Recovery factor (dimensionless, between 0 and 1)
                                  'azimuth_sigma_h_min': 45,  # Azimuth of minimum horizontal stress [degrees]
                                  'l_half_fracture': 150,  # Half-length of hydraulic fracture [m]
                                  },

             'fluid_params': {'pho_surf': 0.841,  # Surface oil density [g/cm³]
                              'mu_o': 0.72,  # Oil viscosity [mPa·s]
                              'mu_w': 0.29,  # Water viscosity [mPa·s]
                              'Bo': 1.241,  # Oil formation volume factor [m³/m³]
                              'Bw': 1,  # Water formation volume factor [m³/m³]
                              },

             'relative_permeability': {'Swc': 0.2,  # Connate water saturation (dimensionless, between 0 and 1)
                                       'Sor': 0.3,  # Residual oil saturation (dimensionless, between 0 and 1)
                                       'Fw': 0.3,  # End-point relative permeability of water
                                       'm1': 1,  # Corey exponent for water phase
                                       'Fo': 1,  # End-point relative permeability of oil
                                       'm2': 1,  # Corey exponent for oil phase
                                       },
             }
