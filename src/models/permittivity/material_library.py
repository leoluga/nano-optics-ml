
MATERIAL_LIBRARY = {
    "LaAlO3": {
        "model": "lorentz_sum",
        "units": "cm^-1",  # Frequencies are in cm^-1
        "data": {
            300: {
                "epsilon_inf": 4.0,
                "oscillators": [
                    {"omega": 184, "S": 14.4,  "gamma": 4.2},
                    {"omega": 428, "S": 4.1,   "gamma": 2.7},
                    {"omega": 496, "S": 0.026, "gamma": 16},
                    {"omega": 652, "S": 0.27,  "gamma": 23.5},
                    {"omega": 692, "S": 0.027, "gamma": 32},
                ],
                "gamma_multiplicator": 2,
            },
            "10,78": {
                "epsilon_inf": 4.0,
                "oscillators": [
                    {"omega": 185, "S": 15,   "gamma": 3.1},
                    {"omega": 427, "S": 4.2,  "gamma": 1.6},
                    {"omega": 498, "S": 0.014,"gamma": 6.5},
                    {"omega": 594, "S": 0.005,"gamma": 7},
                    {"omega": 651, "S": 0.27, "gamma": 15},
                    {"omega": 674, "S": 0.014,"gamma": 49},
                ],
                "gamma_multiplicator": 2,
            }
        },
        "reference":"Zhang1994"
    },
    "LaGaO3": {
        "model": "lorentz_sum",
        "units": "cm^-1",
        "data": {
            300: {
                "epsilon_inf": 4.1,
                "oscillators": [
                    {"omega": 165, "S": 7.7,   "gamma": 2.3},
                    {"omega": 275, "S": 6.7,   "gamma": 8},
                    {"omega": 296, "S": 1.1,   "gamma": 8},
                    {"omega": 308, "S": 2.0,   "gamma": 10},
                    {"omega": 329, "S": 0.64,  "gamma": 8},
                    {"omega": 353, "S": 0.029, "gamma": 4.1},
                    {"omega": 416, "S": 0.013, "gamma": 8},
                    {"omega": 510, "S": 0.038, "gamma": 23},
                    {"omega": 536, "S": 0.032, "gamma": 18},
                    {"omega": 597, "S": 0.25,  "gamma": 30},
                ],
                "gamma_multiplicator": 2,
            }
        },
        "reference":"Zhang1994"
    },
    "NdGaO3": {
        "model": "lorentz_sum",
        "units": "cm^-1",
        "data": {
            300: {
                "epsilon_inf": 4.1,
                "oscillators": [
                    {"omega": 118, "S": 0.28, "gamma": 3.3},
                    {"omega": 174, "S": 5.5,  "gamma": 5.4},
                    {"omega": 244, "S": 0.22, "gamma": 2.7},
                    {"omega": 255, "S": 0.13, "gamma": 4.9},
                    {"omega": 273, "S": 4.7,  "gamma": 6.7},
                    {"omega": 290, "S": 1.2,  "gamma": 6.7},
                    {"omega": 300, "S": 0.53, "gamma": 10.6},
                    {"omega": 321, "S": 1.4,  "gamma": 22},
                    {"omega": 343, "S": 2.1,  "gamma": 9.6},
                    {"omega": 356, "S": 0.28, "gamma": 6.4},
                    {"omega": 424, "S": 0.062,"gamma": 7.8},
                    {"omega": 525, "S": 0.074,"gamma": 32},
                    {"omega": 591, "S": 0.17, "gamma": 33},
                ],
                "gamma_multiplicator": 2,
            }
        },
        "reference":"Zhang1994"
    },
    "STO": {
        "model": "lorentz_product",
        "units": "cm^-1",  # Frequencies are in rad/s
        "data": {
            20: {
                "epsilon_inf": 5.1,
                "modes": [
                    {"omega_L": 169, "gamma_L": 1.9,  "omega_T": 31,  "gamma_T": 1.5},
                    {"omega_L": 475, "gamma_L": 1.9,  "omega_T": 171, "gamma_T": 2.2},
                    {"omega_L": 788, "gamma_L": 18,   "omega_T": 546, "gamma_T": 7.6}
                ]
            },
            300: {
                "epsilon_inf": 5.1,
                "modes": [
                    {"omega_L": 172, "gamma_L": 3.8,  "omega_T": 91,  "gamma_T": 15.0},
                    {"omega_L": 474, "gamma_L": 4.5,  "omega_T": 175, "gamma_T": 5.4},
                    {"omega_L": 788, "gamma_L": 25,   "omega_T": 543, "gamma_T": 17.0}
                ]
            },
        },
        "reference":"Kamaras1995"
    },
    "LSAT": {
        "model": "lorentz_sum",
        "units": "cm^-1",
        "data": {
            # For simplicity, no explicit temperature partition here—just a single dataset:
            None: {
                "epsilon_inf": 4.0,
                "oscillators": [
                    {"omega": 156.9, "A": 6.30,  "gamma": 12.8},
                    {"omega": 222.0, "A": 1.50,  "gamma": 35.0},
                    {"omega": 248.0, "A": 2.60,  "gamma": 42.0},
                    {"omega": 285.9, "A": 4.30,  "gamma": 28.0},
                    {"omega": 330.0, "A": 0.46,  "gamma": 46.0},
                    {"omega": 395.0, "A": 1.89,  "gamma": 44.0},
                    {"omega": 436.4, "A": 0.51,  "gamma": 18.6},
                    {"omega": 659.8, "A": 0.64,  "gamma": 36.5},
                    {"omega": 787.0, "A": 0.0045,"gamma": 26.0},
                ],
                "gamma_multiplicator": 1,
            }
        },
        "reference":"Nunley2016"
    },
    "CaF2":{
        "model": "lorentz_sum",
        "units": "cm^-1",
        "data": {
            # For simplicity, no explicit temperature partition here—just a single dataset:
            None: {
                "epsilon_inf": 2.045,
                "oscillators": [
                    {"omega": 257, "A": 4.2,  "gamma": 1.8},
                    {"omega": 328, "A": 0.4,  "gamma": 35},
                ],
                "gamma_multiplicator": 1,
            }
        },
        "reference":"HandbookOpticalConstants"
    },
    "Gold":{
        "model": "lorentz_sum",
        "units": "cm^-1",
        "data": {
            # For simplicity, no explicit temperature partition here—just a single dataset:
            None: {
                "epsilon_inf": 1,
                "oscillators": [
                    {"omega": (68557**2)/(1e-6**2), "A": 4.2,  "gamma": 387},
                    {"omega": 328, "A": 0.4,  "gamma": 35},
                ],
                "gamma_multiplicator": 1,
            }
        },
        "reference":"gold article"
    },
    "SiO2":{
        "model":"meneses_voigt_model",
        "units": "cm^-1",
        "data": {
            None: {
                "epsilon_inf": 2.1232,
                "oscillators": [
                    {'alpha': 3.7998, 'eta0': 1089.7, 'sigma': 31.454},
                    {'alpha': 0.46089, 'eta0': 1187.7, 'sigma': 100.46},
                    {'alpha': 1.2520, 'eta0': 797.78, 'sigma': 91.601},
                    {'alpha': 7.8147, 'eta0': 1058.2, 'sigma': 63.153},
                    {'alpha': 1.0313, 'eta0': 446.13, 'sigma': 275.111},
                    {'alpha': 5.3757, 'eta0': 443.00, 'sigma': 45.220},
                    {'alpha': 6.3305, 'eta0': 465.80, 'sigma': 22.680},
                    {'alpha': 1.2948, 'eta0': 1026.7, 'sigma': 232.14},
                ],
            }
        },
        "reference":"gold article"
    }
}

def list_available_materials():
    """
    Prints a summary of available materials in the library,
    including their available temperatures, model type, and frequency units.
    """
    print("Available Materials in MATERIAL_LIBRARY:")
    print("-" * 60)
    for material_key, info in MATERIAL_LIBRARY.items():
        model = info.get("model", "Unknown")
        units = info.get("units", "Unknown")
        data = info.get("data", {})
        # The keys in data indicate the available temperatures or conditions.
        available_temps = list(data.keys())
        print(f"Material: {material_key}")
        print(f"  Model: {model}")
        print(f"  Frequency Units: {units}")
        print(f"  Available Temperatures/Conditions: {available_temps}")
        print("-" * 60)

if __name__ == "__main__":
    list_available_materials()