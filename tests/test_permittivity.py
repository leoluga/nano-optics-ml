# tests/test_permittivity.py

import numpy as np
import pytest
from models.permittivity import compute_permittivity, MATERIAL_LIBRARY

def test_laalo3_300K_returns_complex_array():
    """
    Test that LaAlO3 (Zhang1994) at 300 K returns a complex array.
    Frequencies are in cm^-1.
    """
    freq = np.linspace(100, 800, 200)  # cm^-1
    eps = compute_permittivity(freq, "LaAlO3", temperature=300)
    # Check that the output has the same shape as input
    np.testing.assert_array_equal(eps.shape, freq.shape)
    # Check that the result is a complex array
    assert np.iscomplexobj(eps)

def test_sto_300K_returns_complex_array():
    """
    Test that STO (Lorentz product model) at 300 K returns a complex array.
    Frequencies are in rad/s.
    """
    freq = np.linspace(1e11, 1e13, 300)  # rad/s
    eps = compute_permittivity(freq, "STO", temperature=300)
    np.testing.assert_array_equal(eps.shape, freq.shape)
    assert np.iscomplexobj(eps)

def test_lsat_returns_complex_array():
    """
    Test that the Nunley2016 model returns a complex array.
    Frequencies are in cm^-1.
    """
    freq = np.linspace(50, 800, 100)  # cm^-1
    eps = compute_permittivity(freq, "LSAT", temperature=None)
    np.testing.assert_array_equal(eps.shape, freq.shape)
    assert np.iscomplexobj(eps)

def test_invalid_material_raises_error():
    """
    Test that an unknown material key raises a ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        compute_permittivity(100, "NonExistentMaterial", temperature=300)
    assert "Unknown material" in str(excinfo.value)

def test_invalid_temperature_raises_error():
    """
    Test that an invalid temperature raises a ValueError.
    """
    # "LaAlO3_Zhang1994" has data for 300 and "10,78"
    with pytest.raises(ValueError) as excinfo:
        compute_permittivity(100, "LaAlO3", temperature=999)
    assert "not in data" in str(excinfo.value)

def test_default_temperature_error_when_multiple_available():
    """
    Test that if no temperature is provided for a material with multiple datasets,
    a ValueError is raised.
    """
    with pytest.raises(ValueError) as excinfo:
        # LaAlO3_Zhang1994 has multiple temperature datasets.
        compute_permittivity(100, "LaAlO3", temperature=None)
    assert "Must specify temperature" in str(excinfo.value)

def test_material_library_structure():
    """
    Test that the MATERIAL_LIBRARY has the expected keys.
    """
    expected_keys = {"LaAlO3", "STO", "LSAT"}
    assert expected_keys.issubset(set(MATERIAL_LIBRARY.keys()))
