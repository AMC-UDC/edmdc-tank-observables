"""Tests of the observable dictionaries."""

import numpy as np
import pytest

from edmdc_tank_observables.observables.dictionaries import Dictionary, make_dictionary

# Every configuration, with the hyperparameters it needs.
ALL_CONFIGS = [
    ("D0", {}),
    ("D1a", {"order": 3}),
    ("D1b", {"n_centers": 4}),
    ("D1c", {"n_delays": 2}),
    ("D2", {}),
    ("D3a", {"order": 3}),
    ("D3b", {"n_centers": 4}),
    ("D3c", {"n_delays": 2}),
]

# Each combined configuration with the generic one it extends.
COMBINED_PAIRS = [
    ("D1a", "D3a", {"order": 3}),
    ("D1b", "D3b", {"n_centers": 4}),
    ("D1c", "D3c", {"n_delays": 2}),
]

H_SERIES = np.linspace(0.05, 0.95, 12)


def test_d0_returns_the_level_as_a_column():
    """The state-only dictionary returns the level as a single column."""
    psi = make_dictionary("D0").transform(np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(psi, [[0.1], [0.2], [0.3]])


def test_d1a_adds_the_monomials_up_to_the_order():
    """The polynomial dictionary of order 3 gives h, h squared and h cubed."""
    psi = make_dictionary("D1a", order=3).transform(np.array([0.5, 0.2]))
    np.testing.assert_allclose(psi, [[0.5, 0.25, 0.125], [0.2, 0.04, 0.008]])


def test_d1b_centres_are_spread_over_0_1_with_width_equal_to_the_spacing():
    """With 3 centres the bells sit at 0, 0.5 and 1 and are 0.5 wide."""
    psi = make_dictionary("D1b", n_centers=3).transform(np.array([0.5]))
    np.testing.assert_allclose(psi, [[0.5, np.exp(-0.5), 1.0, np.exp(-0.5)]])


def test_d1b_with_a_single_centre_uses_the_width_of_the_range():
    """With one centre at 0 the bell is as wide as the range, which is 1."""
    psi = make_dictionary("D1b", n_centers=1).transform(np.array([0.5]))
    np.testing.assert_allclose(psi, [[0.5, np.exp(-0.5**2 / 2)]])


def test_d1b_accepts_another_range_and_an_explicit_width():
    """A dictionary built by hand can use another range and a given width."""
    psi_range = Dictionary("D1b", None, 1, 0, False, rbf_range=(0.0, 2.0)).transform(np.array([1.0]))
    np.testing.assert_allclose(psi_range, [[1.0, np.exp(-1.0 / (2 * 2.0**2))]])
    psi_width = Dictionary("D1b", None, 3, 0, False, rbf_width=0.25).transform(np.array([0.5]))
    np.testing.assert_allclose(psi_width, [[0.5, np.exp(-0.25 / (2 * 0.25**2)), 1.0, np.exp(-0.25 / (2 * 0.25**2))]])


def test_d1c_stacks_the_delayed_levels_and_trims_the_start():
    """With 2 delays each row is the level followed by the two previous ones."""
    psi = make_dictionary("D1c", n_delays=2).transform(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(psi, [[3, 2, 1], [4, 3, 2], [5, 4, 3]])


def test_align_input_keeps_the_control_of_the_same_instants_as_a_column():
    """The control is trimmed like the observables and returned as a column."""
    u = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(make_dictionary("D1c", n_delays=2).align_input(u), [[30], [40], [50]])
    np.testing.assert_array_equal(make_dictionary("D0").align_input(u).ravel(), u)


def test_d2_adds_the_square_root_of_the_level():
    """The physical dictionary gives the level and its square root."""
    psi = make_dictionary("D2").transform(np.array([0.04, 0.25]))
    np.testing.assert_allclose(psi, [[0.04, 0.2], [0.25, 0.5]])


def test_combined_dictionaries_put_the_square_root_last():
    """In the combined dictionaries the generic columns come first and the square root last."""
    h = np.array([0.04, 0.25])
    np.testing.assert_allclose(make_dictionary("D3a", order=3).transform(h[1:]), [[0.25, 0.0625, 0.015625, 0.5]])
    np.testing.assert_allclose(make_dictionary("D3c", n_delays=1).transform(h), [[0.25, 0.04, 0.5]])


@pytest.mark.parametrize("generic_id, combined_id, kwargs", COMBINED_PAIRS)
def test_combined_equals_the_generic_dictionary_plus_the_square_root(generic_id, combined_id, kwargs):
    """Each combined dictionary is its generic one with the square root appended, on the same rows."""
    generic = make_dictionary(generic_id, **kwargs).transform(H_SERIES)
    combined = make_dictionary(combined_id, **kwargs).transform(H_SERIES)
    n_delays = kwargs.get("n_delays", 0)
    assert combined.shape == (generic.shape[0], generic.shape[1] + 1)
    np.testing.assert_allclose(combined[:, :-1], generic)
    np.testing.assert_allclose(combined[:, -1], np.sqrt(H_SERIES[n_delays:]))


@pytest.mark.parametrize("dict_id, kwargs", ALL_CONFIGS)
def test_dimension_matches_the_number_of_columns(dict_id, kwargs):
    """The announced number of observables equals the columns that transform returns."""
    d = make_dictionary(dict_id, **kwargs)
    assert d.dimension == d.transform(H_SERIES).shape[1]


# Expected observable names of each configuration of ALL_CONFIGS, in column order.
EXPECTED_NAMES = {
    "D0": ["h"],
    "D1a": ["h", "h^2", "h^3"],
    "D1b": ["h", "rbf_1", "rbf_2", "rbf_3", "rbf_4"],
    "D1c": ["h", "h_lag_1", "h_lag_2"],
    "D2": ["h", "sqrt_h"],
    "D3a": ["h", "h^2", "h^3", "sqrt_h"],
    "D3b": ["h", "rbf_1", "rbf_2", "rbf_3", "rbf_4", "sqrt_h"],
    "D3c": ["h", "h_lag_1", "h_lag_2", "sqrt_h"],
}


@pytest.mark.parametrize("dict_id, kwargs", ALL_CONFIGS)
def test_observable_names_give_one_name_per_column_in_order(dict_id, kwargs):
    """Each dictionary names its observables, one per column of transform and in the same order."""
    d = make_dictionary(dict_id, **kwargs)
    assert len(d.observable_names) == d.dimension
    assert d.observable_names == EXPECTED_NAMES[dict_id]


@pytest.mark.parametrize("dict_id, kwargs", ALL_CONFIGS)
def test_first_column_is_always_the_level(dict_id, kwargs):
    """Whatever the dictionary, column 0 is the level itself, trimmed by the delays."""
    d = make_dictionary(dict_id, **kwargs)
    np.testing.assert_array_equal(d.transform(H_SERIES)[:, 0], H_SERIES[d.n_delays:])


@pytest.mark.parametrize("dict_id, kwargs", ALL_CONFIGS)
def test_transform_and_align_input_have_the_same_number_of_rows(dict_id, kwargs):
    """Observables and aligned control describe the same instants, so they have equal length."""
    d = make_dictionary(dict_id, **kwargs)
    assert d.transform(H_SERIES).shape[0] == d.align_input(H_SERIES).shape[0]


@pytest.mark.parametrize("dict_id, kwargs", [("D2", {}), ("D3a", {"order": 2}), ("D3c", {"n_delays": 2})])
def test_negative_level_is_rejected_when_the_square_root_is_used(dict_id, kwargs):
    """A negative level is an error for the dictionaries that take its square root."""
    with pytest.raises(ValueError, match="negative"):
        make_dictionary(dict_id, **kwargs).transform(np.array([0.3, 0.2, -0.01, 0.4]))


def test_negative_level_in_the_trimmed_start_is_also_rejected():
    """The check covers the whole series, including the samples the delays trim away."""
    with pytest.raises(ValueError, match="negative"):
        make_dictionary("D3c", n_delays=2).transform(np.array([-0.01, 0.2, 0.3, 0.4]))


def test_negative_level_is_accepted_without_a_square_root():
    """A dictionary without the square root does not mind a negative level."""
    psi = make_dictionary("D1a", order=2).transform(np.array([-0.1, 0.2]))
    np.testing.assert_allclose(psi, [[-0.1, 0.01], [0.2, 0.04]])


@pytest.mark.parametrize("dict_id", ["D1a", "D1b", "D1c", "D3a", "D3b", "D3c"])
def test_missing_hyperparameter_is_rejected(dict_id):
    """A configuration that needs a hyperparameter fails if it is not given."""
    with pytest.raises(ValueError, match="requires"):
        make_dictionary(dict_id)


@pytest.mark.parametrize(
    "dict_id, kwargs",
    [
        ("D0", {"order": 2}),
        ("D2", {"n_delays": 1}),
        ("D1a", {"order": 2, "n_delays": 1}),
        ("D1b", {"n_centers": 3, "order": 2}),
        ("D1c", {"n_delays": 1, "n_centers": 3}),
    ],
)
def test_unused_hyperparameter_is_rejected(dict_id, kwargs):
    """A hyperparameter that the configuration does not use is an error, not ignored."""
    with pytest.raises(ValueError, match="does not use"):
        make_dictionary(dict_id, **kwargs)


def test_unknown_id_is_rejected():
    """An identifier outside the eight configurations is an error."""
    with pytest.raises(ValueError, match="Unknown dictionary"):
        make_dictionary("D9")


@pytest.mark.parametrize(
    "dict_id, kwargs",
    [
        ("D1a", {"order": 1}),
        ("D3a", {"order": 1}),
        ("D1b", {"n_centers": 0}),
        ("D3b", {"n_centers": 0}),
        ("D1c", {"n_delays": 0}),
        ("D3c", {"n_delays": 0}),
    ],
)
def test_hyperparameter_out_of_range_is_rejected(dict_id, kwargs):
    """An order below 2, no centres, or no delays is rejected."""
    with pytest.raises(ValueError):
        make_dictionary(dict_id, **kwargs)


def test_series_not_longer_than_the_delays_is_rejected():
    """A series with no more samples than delays leaves nothing to transform and is rejected."""
    d = make_dictionary("D1c", n_delays=2)
    with pytest.raises(ValueError, match="samples"):
        d.transform(np.array([0.1, 0.2]))
    assert d.transform(np.array([0.1, 0.2, 0.3])).shape == (1, 3)
