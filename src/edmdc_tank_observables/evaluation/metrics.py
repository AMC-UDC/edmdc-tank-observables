"""Multi-step prediction error on windows anchored at setpoint steps.

All errors are in physical units (% of level).

Windows
    The evaluation segment is scored with one window per step of the setpoint.
    If p is the first sample with the new setpoint, the window starts at
    X = p - 1, the last sample before the change. The model is initialized from
    the measured level up to X, is fed the control that was applied from X
    on, u[X : X + H], and its H predicted levels are compared with the
    measured ones, h[X + 1 : X + 1 + H]. During the horizon the model never
    sees the measured level again.

    An optional offset moves every start by a number of samples. The windows
    depend only on the setpoint of the segment, the horizon and the offset,
    not on the model, so every model is evaluated on exactly the same ones.

Aggregation
    Within a run the errors of all instants of all windows are pooled into one
    RMSE. The summary across runs (median and range) is computed in
    edmdc_tank_observables.evaluation.protocol.
"""

from dataclasses import dataclass

import numpy as np

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.models.edmdc import EDMDcModel


@dataclass
class MultistepError:
    """Result of evaluating a model on a set of windows.

    Attributes
    ----------
    rmse : float
        RMSE over all instants of all windows, in % of level.
    rmse_final : float
        RMSE using only the last instant of each window, in % of level.
    max_abs : float
        Largest absolute error observed, in % of level.
    n_windows : int
        Number of windows evaluated.
    horizon_steps : int
        Length of the horizon, in sampling steps.
    horizon_s : float
        Horizon requested, in seconds. horizon_steps x ts can differ from it
        because the horizon is rounded to whole steps (at least one).
    """

    rmse: float
    rmse_final: float
    max_abs: float
    n_windows: int
    horizon_steps: int
    horizon_s: float


def _horizon_to_steps(horizon_s: float, ts: float) -> int:
    """Convert a horizon in seconds to a whole number of sampling steps (at least 1)."""
    return max(1, int(round(horizon_s / ts)))


def window_starts(
    dataset: PreprocessedDataset, horizon_steps: int, n_delays: int = 0, offset: int = 0
) -> np.ndarray:
    """Start sample X of the window anchored at each setpoint step.

    A step is a sample p whose setpoint differs from the previous one. Its
    window starts at X = p - 1 + offset.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Evaluation segment. Its setpoint (sp) defines the steps.
    horizon_steps : int
        Length of the horizon, in sampling steps.
    n_delays : int, default 0
        Delays of the model to be evaluated: the history of a window is the
        n_delays + 1 samples that end at X.
    offset : int, default 0
        Samples by which every start is moved from X = p - 1, to test how much
        the result depends on where the window starts.

    Returns
    -------
    numpy.ndarray of int
        The start of each window, in increasing order.

    Raises
    ------
    ValueError
        If the setpoint has no steps; if a window starts before the sample
        n_delays, so there is not enough history; if a window does not fit in
        the segment (it needs the samples X to X + horizon_steps); or if
        another setpoint step falls inside a window.
    """
    steps = np.flatnonzero(np.diff(dataset.sp) != 0) + 1
    if len(steps) == 0:
        raise ValueError(f"{dataset.name}: the setpoint has no steps, so there is no window to evaluate.")

    starts = steps - 1 + offset
    for p, x in zip(steps, starts):
        if x < n_delays:
            raise ValueError(
                f"{dataset.name}: the window for the step at sample {p} starts at sample {x}, "
                f"which leaves no room for {n_delays} delays of history."
            )
        if x + horizon_steps + 1 > len(dataset):
            raise ValueError(
                f"{dataset.name}: the window for the step at sample {p} starts at sample {x} "
                f"and does not fit {horizon_steps} steps in {len(dataset)} samples."
            )
        inside = steps[(steps != p) & (steps > x) & (steps <= x + horizon_steps)]
        if len(inside) > 0:
            raise ValueError(
                f"{dataset.name}: another step (sample {inside[0]}) falls inside the window "
                f"for the step at sample {p} ({horizon_steps} steps from sample {x})."
            )
    return starts


def _window_errors(dataset, starts, horizon_steps, model):
    """Error in each window, in % of level (prediction minus measurement).

    Every model goes through the same code and the same windows, so a
    difference in procedure cannot affect the comparison.
    """
    n_delays = model.dictionary.n_delays
    errors = []

    for i in starts:
        history = dataset.h[i - n_delays : i + 1]
        prediction = model.simulate(history, dataset.u[i : i + horizon_steps])
        measured = dataset.h[i + 1 : i + 1 + horizon_steps]
        errors.append(dataset.denormalize_h_delta(prediction - measured))

    return errors


def _summarize(errors, horizon_steps, horizon_s):
    """Pool the errors of a set of windows into a MultistepError."""
    all_errors = np.concatenate(errors)
    final_errors = np.array([e[-1] for e in errors])
    return MultistepError(
        rmse=float(np.sqrt(np.mean(all_errors**2))),
        rmse_final=float(np.sqrt(np.mean(final_errors**2))),
        max_abs=float(np.max(np.abs(all_errors))),
        n_windows=len(errors),
        horizon_steps=horizon_steps,
        horizon_s=horizon_s,
    )


def multistep_error(
    dataset: PreprocessedDataset,
    horizon_s: float,
    model: EDMDcModel,
    offset: int = 0,
) -> MultistepError:
    """Multi-step error of a model on the anchored windows.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Evaluation segment.
    horizon_s : float
        Length of the horizon, in seconds. It is converted to steps with the
        sampling period of the dataset.
    model : EDMDcModel
        Model to evaluate.
    offset : int, default 0
        Samples by which every window start is moved; see :func:`window_starts`.

    Returns
    -------
    MultistepError
        The pooled error over the windows, in % of level.

    Raises
    ------
    ValueError
        If the windows are not valid; see :func:`window_starts`.
    """
    n_delays = model.dictionary.n_delays
    horizon_steps = _horizon_to_steps(horizon_s, dataset.ts)
    starts = window_starts(dataset, horizon_steps, n_delays, offset)
    errors = _window_errors(dataset, starts, horizon_steps, model)
    return _summarize(errors, horizon_steps, horizon_s)
