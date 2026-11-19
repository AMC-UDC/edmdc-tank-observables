"""Tests of the static curve of the plant on synthetic runs whose answers can be worked out by hand."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/plant_static_curve.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "plant_static_curve.py"
    spec = importlib.util.spec_from_file_location("plant_static_curve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


psc = load_script()
STEP = psc.STEP

# Synthetic run: five plateaus of 400 s each, setpoints 10 to 50 %.
PLATEAU_S = 400
SETPOINTS = (10.0, 20.0, 30.0, 40.0, 50.0)


def staircase_run(level_in_window, control=None):
    """One run at 1 Hz with a setpoint staircase.

    Each plateau starts with 100 s at setpoint - 5 (outside the last-300-s
    window) and then follows level_in_window(sp, t_rel), with t_rel the time in
    s from the start of the window. Control is 2 x setpoint + an alternation of
    +-1 unless given.
    """
    rows = []
    for k, sp in enumerate(SETPOINTS):
        for j in range(PLATEAU_S):
            t = k * PLATEAU_S + j
            t_rel = j - (PLATEAU_S - psc.WINDOW_S)
            h = sp - 5.0 if t_rel < 0 else level_in_window(sp, t_rel)
            u = 2.0 * sp + (1.0 if j % 2 == 0 else -1.0) if control is None else control(sp, j)
            rows.append({"t_s": float(t), "Setpoint_pct": sp, "Level_pct": h, "Control_Signal_pct": u})
    return pd.DataFrame(rows)


def alternating(sp, t_rel):
    """Level that alternates sp + 0.1 and sp - 0.1: mean sp, standard deviation 0.1."""
    return sp + (0.1 if t_rel % 2 == 0 else -0.1)


def test_plateau_windows_means_and_deviations():
    """Each plateau is summarized over its last 300 s: mean and standard deviation of level and control."""
    plateaus = psc.plateau_table(staircase_run(alternating), "r", 50.0)
    assert len(plateaus) == 5
    assert plateaus["setpoint_pct"].tolist() == list(SETPOINTS)
    assert plateaus["window_start_s"].tolist() == [100.0, 500.0, 900.0, 1300.0, 1700.0]
    assert plateaus["window_end_s"].tolist() == [400.0, 800.0, 1200.0, 1600.0, 2000.0]
    assert (plateaus["n_samples"] == 300).all()
    np.testing.assert_allclose(plateaus["h_ss_pct"], SETPOINTS)
    np.testing.assert_allclose(plateaus["h_std_pct"], 0.1)
    np.testing.assert_allclose(plateaus["u_ss_pct"], 2.0 * np.array(SETPOINTS))
    np.testing.assert_allclose(plateaus["u_std_pct"], 1.0)


def test_last_plateau_ends_with_the_run():
    """The window of the last plateau ends one second after the last sample of the run."""
    plateaus = psc.plateau_table(staircase_run(alternating), "r", 50.0)
    last = plateaus.iloc[-1]
    assert last["window_end_s"] == 5 * PLATEAU_S
    assert last["window_start_s"] == 5 * PLATEAU_S - 300


def test_missing_sample_shortens_the_window_but_not_its_limits():
    """A sample dropped inside a window is just left out: the window keeps its limits in time."""
    df = staircase_run(alternating)
    df = df[df["t_s"] != 250.0].reset_index(drop=True)
    first = psc.plateau_table(df, "r", 50.0).iloc[0]
    assert first["n_samples"] == 299
    assert (first["window_start_s"], first["window_end_s"]) == (100.0, 400.0)


def ramp_run(drift):
    """Run whose level rises linearly across each window by `drift` over 300 s, centred on the setpoint."""
    return staircase_run(lambda sp, t_rel: sp + drift / 300.0 * (t_rel - 149.5))


def test_drift_of_a_known_ramp():
    """A level that rises 0.3 points over the window has a drift of 0.3 and no offset."""
    plateaus = psc.plateau_table(ramp_run(0.3), "r", 50.0)
    np.testing.assert_allclose(plateaus["drift_pct"], 0.3)
    np.testing.assert_allclose(plateaus["offset_pct"], 0.0, atol=1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stationary_flag_by_drift(sign):
    """A drift just below one recording step is stationary; just above it is not, in either direction."""
    below = psc.plateau_table(ramp_run(sign * 0.99 * STEP), "r", 50.0)
    above = psc.plateau_table(ramp_run(sign * 1.01 * STEP), "r", 50.0)
    assert below["stationary"].all()
    assert not above["stationary"].any()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stationary_flag_by_offset(sign):
    """A constant level just under one recording step away from the setpoint is stationary; just over it is not."""
    below = psc.plateau_table(staircase_run(lambda sp, t: sp + sign * 0.99 * STEP), "r", 50.0)
    above = psc.plateau_table(staircase_run(lambda sp, t: sp + sign * 1.01 * STEP), "r", 50.0)
    np.testing.assert_allclose(below["drift_pct"], 0.0, atol=1e-9)
    assert below["stationary"].all()
    assert not above["stationary"].any()


X = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.9])


@pytest.mark.parametrize(
    "shape, curvature",
    [("sqrt", np.sqrt(X)), ("quadratic", X**2)],
)
def test_exact_curve_is_recovered_and_the_line_is_not(shape, curvature):
    """Data that follow u = a + b x + c g(x) exactly give back a, b, c with zero error; a straight line does not fit them."""
    a, b, c = 5.0, 20.0, 40.0
    u = a + b * X + c * curvature
    coefs, rmse_in, rmse_loo = psc.fit_shape(X, u, shape)
    np.testing.assert_allclose(coefs, [a, b, c], rtol=1e-9)
    assert rmse_in < 1e-9 and rmse_loo < 1e-9
    _, line_in, line_loo = psc.fit_shape(X, u, "linear")
    assert line_in > 1e-3 and line_loo > 1e-3


def test_leave_one_out_by_hand():
    """Straight line on (1, 1), (2, 2), (3, 3), (4, 5): leaving each point out, the misses are 2/3, -1/7, -4/7 and 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    u = np.array([1.0, 2.0, 3.0, 5.0])
    misses = np.array([2 / 3, -1 / 7, -4 / 7, 1.0])
    _, rmse_in, rmse_loo = psc.fit_shape(x, u, "linear")
    assert rmse_loo == pytest.approx(np.sqrt(np.mean(misses**2)))
    # In-sample line: u = -0.5 + 1.3 x, residuals 0.2, -0.1, -0.4, 0.3.
    assert rmse_in == pytest.approx(np.sqrt(np.mean(np.array([0.2, -0.1, -0.4, 0.3]) ** 2)))


def plateau_rows(h_values, stationary=True):
    """Plateaus table of one run with the given levels and control u = h / 2."""
    return pd.DataFrame(
        {
            "run": "r",
            "valve_opening_pct": 50.0,
            "h_ss_pct": h_values,
            "u_ss_pct": np.asarray(h_values) / 2.0,
            "stationary": stationary,
        }
    )


def test_curve_fits_need_five_plateaus():
    """A run with fewer than five plateaus in the variant is rejected."""
    plateaus = plateau_rows([10.0, 20.0, 30.0, 40.0, 50.0], stationary=[True, True, True, True, False])
    psc.curve_fits(plateaus, "all_plateaus")  # five plateaus: accepted
    with pytest.raises(ValueError, match="fewer than 5"):
        psc.curve_fits(plateaus, "stationary_only")


def test_curve_fits_reject_a_non_positive_level():
    """A plateau with a level of zero or below is rejected, since sqrt(x) needs x > 0."""
    with pytest.raises(ValueError, match="h_ss"):
        psc.curve_fits(plateau_rows([0.0, 20.0, 30.0, 40.0, 50.0]), "all_plateaus")


def first_order_run(tau, sp_values=(10.0, 20.0, 30.0, 40.0, 50.0)):
    """Run whose level follows a first-order response with time constant tau after each setpoint step."""
    rows = []
    for k, sp in enumerate(sp_values):
        sp_before = sp_values[k - 1] if k > 0 else sp
        for j in range(PLATEAU_S):
            h = sp_before + (sp - sp_before) * (1.0 - np.exp(-j / tau))
            rows.append({"t_s": float(k * PLATEAU_S + j), "Setpoint_pct": sp, "Level_pct": h, "Control_Signal_pct": 0.0})
    return pd.DataFrame(rows)


def test_t95_of_a_first_order_response():
    """With tau = 20 s the level reaches 95 % of the step after 60 s (20 ln 20 = 59.9 s, first whole second)."""
    steps = psc.step_table(first_order_run(20.0), "r", 50.0, cut_time_s=1000.0)
    assert len(steps) == 4
    assert steps["t_change_s"].tolist() == [400.0, 800.0, 1200.0, 1600.0]
    assert (steps["t95_s"] == 60.0).all()
    assert steps["is_evaluation_step"].tolist() == [False, False, True, True]


def test_t95_is_nan_when_not_reached_within_the_plateau():
    """With tau = 1000 s the level does not reach 95 % of the step within the 400 s of the new plateau."""
    steps = psc.step_table(first_order_run(1000.0), "r", 50.0, cut_time_s=1000.0)
    assert steps["t95_s"].isna().all()


def test_t95_rejects_a_descending_step():
    """A setpoint step downwards is rejected."""
    with pytest.raises(ValueError, match="not ascending"):
        psc.step_table(first_order_run(20.0, (10.0, 20.0, 15.0)), "r", 50.0, cut_time_s=1000.0)


def test_figure_needs_one_run_per_panel_of_the_grid():
    """With 9 runs for the 2 x 5 grid of panels the figure is not drawn and the error names both numbers."""
    plateaus = pd.DataFrame({"run": [f"r{i}" for i in range(9)]})
    with pytest.raises(ValueError, match="2 x 5 grid.*9 runs"):
        psc.plot_curves(plateaus, pd.DataFrame())
