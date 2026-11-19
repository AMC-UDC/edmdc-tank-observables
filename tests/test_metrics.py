"""Tests of the evaluation windows and of the error metrics."""

import numpy as np
import pytest

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.evaluation import metrics
from edmdc_tank_observables.evaluation.metrics import (
    MultistepError,
    _summarize,
    multistep_error,
    window_starts,
)
from edmdc_tank_observables.models.edmdc import EDMDcModel
from edmdc_tank_observables.observables.dictionaries import make_dictionary


def make_dataset(h, u, sp, h_scale=(0.0, 100.0)):
    """Build a dataset sampled every 5 s from level, control and setpoint series."""
    h = np.asarray(h, dtype=float)
    return PreprocessedDataset(
        h=h, u=np.asarray(u, dtype=float), sp=np.asarray(sp, dtype=float),
        t=5.0 * np.arange(len(h)), ts=5.0, h_scale=h_scale, u_scale=(0.0, 100.0), name="synthetic",
    )


def staircase(n, step_samples, base=0.2, jump=0.1):
    """A setpoint that rises by `jump` at each of the given samples (the first sample with the new value)."""
    sp = np.full(n, base)
    for p in step_samples:
        sp[p:] += jump
    return sp


def ramp_dataset(n=40, step_samples=(10, 20, 30), h_scale=(0.0, 100.0)):
    """A level that rises 0.01 per sample, with a staircase setpoint."""
    return make_dataset(0.01 * np.arange(n), np.zeros(n), staircase(n, step_samples), h_scale)


def persistence_model():
    """A state-only model that keeps the level where it is."""
    return EDMDcModel(A=np.array([[1.0]]), B=np.array([[0.0]]), c=np.array([0.0]),
                      dictionary=make_dictionary("D0"))


def three_delay_model():
    """A model with 3 delays, so its windows need 3 samples of history."""
    shift = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    return EDMDcModel(A=shift, B=np.zeros((4, 1)), c=np.zeros(4),
                      dictionary=make_dictionary("D1c", n_delays=3))


def test_windows_start_at_the_sample_before_each_setpoint_step():
    """With steps at samples 10, 20 and 30 the windows start at 9, 19 and 29."""
    starts = window_starts(ramp_dataset(), horizon_steps=5)
    np.testing.assert_array_equal(starts, [9, 19, 29])


def test_offset_moves_every_start_by_the_same_number_of_samples():
    """A positive or negative offset shifts all the starts together."""
    ds = ramp_dataset()
    np.testing.assert_array_equal(window_starts(ds, 5, offset=2), [11, 21, 31])
    np.testing.assert_array_equal(window_starts(ds, 5, offset=-1), [8, 18, 28])


def test_starts_do_not_depend_on_the_delays_of_the_model():
    """As long as the windows are valid, a model with delays gets the same starts as one without."""
    ds = ramp_dataset()
    np.testing.assert_array_equal(window_starts(ds, 5, n_delays=3), window_starts(ds, 5, n_delays=0))


def test_setpoint_without_steps_is_rejected():
    """If the setpoint never changes there is nothing to anchor a window to, and that is an error."""
    ds = make_dataset(np.zeros(40), np.zeros(40), np.full(40, 0.5))
    with pytest.raises(ValueError, match="no steps"):
        window_starts(ds, 5)


def test_window_that_would_need_more_history_than_exists_is_rejected():
    """A step so early that the model's delays do not fit before it is an error."""
    ds = ramp_dataset(step_samples=(3, 20))  # first window starts at 2
    window_starts(ds, 5, n_delays=2)
    with pytest.raises(ValueError, match="history"):
        window_starts(ds, 5, n_delays=3)


def test_window_that_runs_past_the_end_is_rejected():
    """A step so late that the whole horizon does not fit in the segment is an error."""
    ds = ramp_dataset(n=40, step_samples=(10, 34))  # last window starts at 33; with 6 steps it ends at 39
    window_starts(ds, 6)
    with pytest.raises(ValueError, match="fit"):
        window_starts(ds, 7)


def test_another_step_inside_the_window_is_rejected():
    """If a second setpoint step happens within the horizon of a window, the window is rejected."""
    ds = ramp_dataset(step_samples=(10, 13))
    window_starts(ds, 2)
    with pytest.raises(ValueError, match="another step"):
        window_starts(ds, 5)


def test_offset_that_pushes_a_window_out_of_range_is_rejected():
    """A shift that makes a window start before the history or end after the data is an error."""
    ds = ramp_dataset()
    with pytest.raises(ValueError):
        window_starts(ds, 5, offset=-10)
    with pytest.raises(ValueError):
        window_starts(ds, 5, offset=20)


def test_exact_linear_model_has_zero_error():
    """A model that is exactly the system that produced the data predicts every window without error."""
    n = 40
    u = 0.5 + 0.3 * np.sin(0.4 * np.arange(n))
    h = [0.3]
    for k in range(n - 1):
        h.append(0.9 * h[-1] + 0.2 * u[k] + 0.05)
    model = EDMDcModel(A=np.array([[0.9]]), B=np.array([[0.2]]), c=np.array([0.05]),
                       dictionary=make_dictionary("D0"))
    result = multistep_error(make_dataset(h, u, staircase(n, (10, 20, 30))), horizon_s=25, model=model)
    assert result.n_windows == 3
    assert result.rmse == pytest.approx(0.0, abs=1e-10)
    assert result.max_abs == pytest.approx(0.0, abs=1e-10)


def test_constant_level_model_on_a_ramp_has_the_expected_error():
    """Repeating the level at the start of the window on a ramp of 1 % per sample gives errors of -1, -2, -3, -4 and -5 %."""
    result = multistep_error(ramp_dataset(), horizon_s=25, model=persistence_model())  # 5 steps
    assert result.n_windows == 3
    assert result.horizon_steps == 5
    assert result.horizon_s == 25
    assert result.rmse == pytest.approx(np.sqrt(11.0))  # sqrt((1 + 4 + 9 + 16 + 25) / 5)
    assert result.rmse_final == pytest.approx(5.0)
    assert result.max_abs == pytest.approx(5.0)


def test_error_is_converted_to_percent_with_the_level_scale():
    """With a level range of 50 % instead of 100 %, the same normalized error counts for half."""
    wide = multistep_error(ramp_dataset(h_scale=(0.0, 100.0)), horizon_s=25, model=persistence_model())
    narrow = multistep_error(ramp_dataset(h_scale=(20.0, 70.0)), horizon_s=25, model=persistence_model())
    assert narrow.rmse == pytest.approx(wide.rmse / 2)


def test_horizon_in_seconds_is_converted_to_steps_with_the_sampling_period():
    """At 5 s per sample, a horizon of 30 s is 6 steps."""
    result = multistep_error(ramp_dataset(), horizon_s=30, model=persistence_model())
    assert (result.horizon_steps, result.horizon_s) == (6, 30)


def test_control_of_each_window_starts_at_the_last_sample_of_the_history():
    """The control used at step j is the one applied at the instant of the last history sample plus j."""
    n = 40
    u = np.random.default_rng(0).uniform(0.2, 0.8, n)
    h = np.concatenate([[0.1], u[:-1]])  # h[k + 1] = u[k]
    model = EDMDcModel(A=np.array([[0.0]]), B=np.array([[1.0]]), c=np.array([0.0]),
                       dictionary=make_dictionary("D0"))  # predicts the next level as the current control
    result = multistep_error(make_dataset(h, u, staircase(n, (10, 20, 30))), horizon_s=25, model=model)
    assert result.max_abs == pytest.approx(0.0, abs=1e-10)


def test_history_of_a_model_with_delays_ends_at_the_start_of_the_window():
    """For a model with one delay the history is the level at the start of the window and the one before."""
    # The model swaps [h_k, h_{k-1}], so it predicts h_{X-1}, h_X, h_{X-1}, h_X, h_{X-1} against the ramp.
    swap = EDMDcModel(A=np.array([[0.0, 1.0], [1.0, 0.0]]), B=np.zeros((2, 1)), c=np.zeros(2),
                      dictionary=make_dictionary("D1c", n_delays=1))
    result = multistep_error(ramp_dataset(), horizon_s=25, model=swap)
    # Errors in % of level are -2, -2, -4, -4, -6.
    assert result.rmse == pytest.approx(np.sqrt((4 + 4 + 16 + 16 + 36) / 5))
    assert result.rmse_final == pytest.approx(6.0)
    assert result.max_abs == pytest.approx(6.0)


def test_constant_level_model_is_compared_with_the_samples_after_the_window_start():
    """A model that keeps the level predicts h[X] at every step, and its error matches one computed by hand from h[X + 1 : X + 6]."""
    n = 40
    h = np.random.default_rng(1).uniform(0.2, 0.8, n)
    u = np.random.default_rng(2).uniform(0.2, 0.8, n)
    ds = make_dataset(h, u, staircase(n, (10, 20, 30)))
    model = persistence_model()
    starts = (9, 19, 29)  # the sample before each step
    for x in starts:
        np.testing.assert_allclose(model.simulate(h[x : x + 1], u[x : x + 5]), np.full(5, h[x]))

    by_hand = [100.0 * (h[x] - h[x + 1 : x + 6]) for x in starts]  # prediction minus measurement, in %
    result = multistep_error(ds, horizon_s=25, model=model)
    pooled = np.concatenate(by_hand)
    assert result.n_windows == 3
    assert result.rmse == pytest.approx(np.sqrt(np.mean(pooled**2)))
    assert result.rmse_final == pytest.approx(np.sqrt(np.mean([e[-1] ** 2 for e in by_hand])))
    assert result.max_abs == pytest.approx(np.max(np.abs(pooled)))


def test_models_with_and_without_delays_use_the_same_windows(monkeypatch):
    """The state-only model and a model with 3 delays are evaluated on identical starts."""
    seen = []
    original = metrics._window_errors

    def spy(dataset, starts, horizon_steps, model):
        seen.append(list(starts))
        return original(dataset, starts, horizon_steps, model)

    monkeypatch.setattr(metrics, "_window_errors", spy)
    ds = ramp_dataset()
    multistep_error(ds, 25, model=persistence_model())
    multistep_error(ds, 25, model=three_delay_model())
    assert seen[0] == seen[1] == [9, 19, 29]


def test_pooled_error_is_computed_over_all_instants_of_all_windows():
    """The RMSE pools every instant of every window; the final RMSE and the maximum use the last instant and the largest value."""
    errors = [np.array([0.0, -2.0]), np.array([4.0, 4.0])]
    result = _summarize(errors, horizon_steps=2, horizon_s=10.0)
    assert result.rmse == pytest.approx(np.sqrt((0 + 4 + 16 + 16) / 4))
    assert result.rmse_final == pytest.approx(np.sqrt((4 + 16) / 2))
    assert result.max_abs == pytest.approx(4.0)
    assert (result.n_windows, result.horizon_steps, result.horizon_s) == (2, 2, 10.0)
    assert isinstance(result, MultistepError)
