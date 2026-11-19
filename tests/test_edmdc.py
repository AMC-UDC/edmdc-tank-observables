"""Tests of the EDMDc model: fitting, one-step prediction and open-loop simulation."""

import numpy as np
import pytest

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.models.edmdc import EDMDcModel, fit_edmdc
from edmdc_tank_observables.observables.dictionaries import make_dictionary


def make_dataset(h, u, h_scale=(0.0, 100.0)):
    """Wrap level and control series in a dataset sampled every 5 s."""
    h, u = np.asarray(h, dtype=float), np.asarray(u, dtype=float)
    return PreprocessedDataset(
        h=h, u=u, sp=np.zeros_like(h), t=5.0 * np.arange(len(h)), ts=5.0,
        h_scale=h_scale, u_scale=(0.0, 100.0), name="synthetic",
    )


def varying_control(n):
    """A control signal that is not constant, so the input can be told apart from the constant term."""
    k = np.arange(n)
    return 0.5 + 0.3 * np.sin(0.4 * k) + 0.1 * np.cos(1.3 * k)


def run_system(step, h0, u):
    """Generate a level series h_{k+1} = step(h_k, u_k) from h0 under the control u."""
    h = [h0]
    for u_k in u[:-1]:
        h.append(step(h[-1], u_k))
    return np.array(h)


def noisy_ar_run(n_lags, n=400, seed=0):
    """A noisy level series driven by its own past (n_lags terms) and a random control."""
    rng = np.random.default_rng(seed)
    coefs = [0.5, 0.2, 0.1, 0.05][:n_lags]
    u = rng.uniform(0.2, 0.8, n)
    h = list(rng.uniform(0.3, 0.5, n_lags))
    for k in range(n_lags - 1, n - 1):
        past = sum(c * h[k - j] for j, c in enumerate(coefs))
        h.append(past + 0.1 * u[k] + 0.02 + 0.01 * rng.standard_normal())
    return np.array(h), u


def hand_built_d2_model():
    """A D2 model with known matrices whose second state does not follow sqrt of the first."""
    return EDMDcModel(
        A=np.array([[0.5, 0.2], [0.1, 0.3]]),
        B=np.array([[0.1], [0.2]]),
        c=np.array([0.01, 0.02]),
        dictionary=make_dictionary("D2"),
    )


def test_known_linear_system_is_recovered_by_d0():
    """On data from h' = 0.9 h + 0.2 u + 0.05 the state-only model finds exactly those numbers."""
    u = varying_control(80)
    h = run_system(lambda h, u: 0.9 * h + 0.2 * u + 0.05, 0.2, u)
    model = fit_edmdc(make_dictionary("D0"), make_dataset(h, u))
    np.testing.assert_allclose(model.A, [[0.9]], atol=1e-10)
    np.testing.assert_allclose(model.B, [[0.2]], atol=1e-10)
    np.testing.assert_allclose(model.c, [0.05], atol=1e-10)
    assert model.fit_rmse_pct == pytest.approx(0.0, abs=1e-8)


def test_fit_records_the_sampling_period_pairs_and_conditioning():
    """The fitted model keeps the sampling period, the number of pairs used and a finite condition number."""
    u = varying_control(80)
    h = run_system(lambda h, u: 0.9 * h + 0.2 * u + 0.05, 0.2, u)
    model = fit_edmdc(make_dictionary("D0"), make_dataset(h, u))
    assert model.ts == 5.0
    assert model.n_pairs == 79
    assert model.p == 1
    assert 1.0 <= model.cond < 1e6


def test_d2_recovers_the_coefficients_of_a_system_with_a_square_root_term():
    """On data from h' = a h + b sqrt(h) + d u + e, the first row of the model finds a, b, d and e."""
    a, b, d, e = 0.7, 0.15, 0.1, 0.02
    u = varying_control(120)
    h = run_system(lambda h, u: a * h + b * np.sqrt(h) + d * u + e, 0.3, u)
    model = fit_edmdc(make_dictionary("D2"), make_dataset(h, u))
    np.testing.assert_allclose(model.A[0], [a, b], atol=1e-8)
    np.testing.assert_allclose(model.B[0, 0], d, atol=1e-8)
    np.testing.assert_allclose(model.c[0], e, atol=1e-8)


def test_fit_rmse_is_the_one_step_error_of_the_level_in_percent():
    """The reported fit error is the one-step error of the level, converted to % with the level scale."""
    h, u = noisy_ar_run(1)
    model = fit_edmdc(make_dictionary("D0"), make_dataset(h, u, h_scale=(0.0, 100.0)))
    predicted = model.A[0, 0] * h[:-1] + model.B[0, 0] * u[:-1] + model.c[0]
    expected = 100.0 * np.sqrt(np.mean((h[1:] - predicted) ** 2))
    assert model.fit_rmse_pct == pytest.approx(expected, rel=1e-9)


def test_fit_rmse_uses_the_range_of_the_level_scale():
    """With a level range of 50 % instead of 100 %, the same normalized error counts for half."""
    h, u = noisy_ar_run(1)
    wide = fit_edmdc(make_dictionary("D0"), make_dataset(h, u, h_scale=(0.0, 100.0)))
    narrow = fit_edmdc(make_dictionary("D0"), make_dataset(h, u, h_scale=(20.0, 70.0)))
    assert narrow.fit_rmse_pct == pytest.approx(wide.fit_rmse_pct / 2, rel=1e-9)


def test_spectral_radius_is_the_largest_eigenvalue_modulus():
    """The spectral radius is the largest modulus among the eigenvalues of A."""
    rotation = EDMDcModel(
        A=np.array([[0.0, -0.5], [0.5, 0.0]]), B=np.zeros((2, 1)), c=np.zeros(2),
        dictionary=make_dictionary("D2"),
    )
    assert rotation.spectral_radius == pytest.approx(0.5)
    diagonal = EDMDcModel(
        A=np.diag([0.9, -0.95]), B=np.zeros((2, 1)), c=np.zeros(2), dictionary=make_dictionary("D2")
    )
    assert diagonal.spectral_radius == pytest.approx(0.95)


def test_simulate_returns_the_first_component_of_the_lifted_state():
    """The simulated level is the first component of the lifted state, not the second one."""
    model = hand_built_d2_model()
    # Start from h = 0.25, so the lifted state is [0.25, 0.5].
    # Step 1 (u = 1): [0.5*0.25 + 0.2*0.5 + 0.1 + 0.01, 0.1*0.25 + 0.3*0.5 + 0.2 + 0.02] = [0.335, 0.395].
    # Step 2 (u = 0): first component 0.5*0.335 + 0.2*0.395 + 0.01 = 0.2565.
    prediction = model.simulate(np.array([0.25]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(prediction, [0.335, 0.2565])


def test_simulate_for_one_step_matches_predict_step():
    """A one-step simulation gives the first component of predict_step from the lifted last level."""
    model = hand_built_d2_model()
    h_history = np.array([0.25])
    psi = model.dictionary.transform(h_history)[-1]
    expected = model.predict_step(psi[None, :], np.array([[0.7]]))[0, 0]
    np.testing.assert_allclose(model.simulate(h_history, np.array([0.7])), [expected])


def test_simulate_equals_iterating_predict_step():
    """A long simulation equals applying predict_step again and again to the lifted state."""
    model = hand_built_d2_model()
    u = np.array([0.3, 0.9, 0.1, 0.5, 0.5])
    psi = model.dictionary.transform(np.array([0.36]))[-1][None, :]
    expected = []
    for u_k in u:
        psi = model.predict_step(psi, np.array([[u_k]]))
        expected.append(psi[0, 0])
    np.testing.assert_allclose(model.simulate(np.array([0.36]), u), expected)


def test_simulate_propagates_the_lifted_state_linearly_without_recomputing_the_square_root():
    """After the first step the second state is not replaced by sqrt of the predicted level."""
    model = hand_built_d2_model()
    prediction = model.simulate(np.array([0.25]), np.array([1.0, 0.0]))
    # Linear recursion by hand: second component after step 1 is 0.395, not sqrt(0.335).
    linear_step_2 = 0.5 * 0.335 + 0.2 * 0.395 + 0.01
    recomputed_step_2 = 0.5 * 0.335 + 0.2 * np.sqrt(0.335) + 0.01
    assert prediction[1] == pytest.approx(linear_step_2)
    assert prediction[1] != pytest.approx(recomputed_step_2, abs=1e-3)


def test_simulate_uses_the_last_samples_of_a_longer_history():
    """When the history is longer than needed, only its last samples are used."""
    model = hand_built_d2_model()
    u = np.array([0.4, 0.4])
    np.testing.assert_array_equal(
        model.simulate(np.array([0.9, 0.1, 0.25]), u), model.simulate(np.array([0.25]), u)
    )


def test_d0_is_an_arx_with_one_term_and_a_constant():
    """The state-only model has the same coefficients as a separately fitted ARX(1,1) with constant."""
    h, u = noisy_ar_run(1)
    model = fit_edmdc(make_dictionary("D0"), make_dataset(h, u))
    regressors = np.column_stack([h[:-1], u[:-1], np.ones(len(h) - 1)])
    arx, *_ = np.linalg.lstsq(regressors, h[1:], rcond=None)
    np.testing.assert_allclose(
        [model.A[0, 0], model.B[0, 0], model.c[0]], arx, rtol=1e-9, atol=1e-12
    )


def test_d1c_is_an_arx_over_the_last_four_levels_and_shifts_the_delays_exactly():
    """With 3 delays the first row is an ARX over 4 past levels, and the other rows just shift the levels."""
    h, u = noisy_ar_run(4)
    model = fit_edmdc(make_dictionary("D1c", n_delays=3), make_dataset(h, u))
    n = len(h)
    # Same pairs as the dictionary leaves: current time k = 3 .. n - 2, target h_{k+1}.
    regressors = np.column_stack(
        [h[3 : n - 1], h[2 : n - 2], h[1 : n - 3], h[0 : n - 4], u[3 : n - 1], np.ones(n - 4)]
    )
    arx, *_ = np.linalg.lstsq(regressors, h[4:], rcond=None)
    np.testing.assert_allclose(model.A[0], arx[:4], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(model.B[0, 0], arx[4], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(model.c[0], arx[5], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(model.A[1:], np.eye(3, 4), atol=1e-9)
    np.testing.assert_allclose(model.B[1:], 0.0, atol=1e-9)
    np.testing.assert_allclose(model.c[1:], 0.0, atol=1e-9)


def test_d1c_simulation_matches_the_open_loop_arx_recursion():
    """With 3 delays the simulation equals running the 4-term ARX forward on its own predictions."""
    h, u = noisy_ar_run(4)
    model = fit_edmdc(make_dictionary("D1c", n_delays=3), make_dataset(h, u))
    weights, b, c = model.A[0], model.B[0, 0], model.c[0]
    u_future = u[10:30]
    levels = list(h[7:11])  # four consecutive levels, the last one at the time of u_future[0]
    expected = []
    for u_k in u_future:
        next_level = weights @ np.array(levels[::-1][:4]) + b * u_k + c
        levels.append(next_level)
        expected.append(next_level)
    np.testing.assert_allclose(model.simulate(h[7:11], u_future), expected, rtol=1e-9, atol=1e-9)


def test_fit_with_fewer_pairs_than_regressors_is_rejected():
    """A series that gives fewer pairs than columns of the regression fails with a clear error."""
    h, u = [0.1, 0.2, 0.4], [0.5, 0.1, 0.7]  # 2 pairs, 3 regressors
    with pytest.raises(ValueError, match="pairs"):
        fit_edmdc(make_dictionary("D0"), make_dataset(h, u))


def test_fit_with_as_many_pairs_as_regressors_is_accepted():
    """With exactly as many pairs as regressors the fit still runs."""
    h, u = [0.1, 0.2, 0.4, 0.3], [0.5, 0.1, 0.7, 0.2]  # 3 pairs, 3 regressors
    assert fit_edmdc(make_dictionary("D0"), make_dataset(h, u)).n_pairs == 3


def test_simulate_with_too_short_a_history_is_rejected():
    """Simulating needs one more level than the number of delays, and fails otherwise."""
    h, u = noisy_ar_run(4)
    model = fit_edmdc(make_dictionary("D1c", n_delays=3), make_dataset(h, u))
    with pytest.raises(ValueError, match="history"):
        model.simulate(h[:3], np.array([0.5, 0.5]))
