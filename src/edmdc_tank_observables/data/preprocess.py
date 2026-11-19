"""Preprocessing of the Baseline runs.

Resampling
    The acquisition (1 Hz) is heavily oversampled with respect to the tank
    dynamics, so each run is decimated to a sampling period of several
    seconds by keeping the samples whose time is a multiple of that period (no
    averaging).

Gaps
    Dropping sensor faults opens gaps in the series. A pair of samples
    (k, k+1) that crosses a gap is not a real transition of the system.
    Gaps are not handled: if any remains after resampling, preprocessing fails.

Normalization
    The level h and the control signal u are scaled to [0, 1] with their
    physical range, 0-100 % (LEVEL_RANGE and CONTROL_RANGE). The scale is not
    estimated from the data.

Split
    The fit/evaluation split is a temporal cut, not a random one. It is made
    at an absolute instant, a fraction of the acquisition time, and not at a
    number of samples: the runs do not all start on the same sample of the
    grid (rows dropped at the start move the first one by a few seconds), but
    they end at the same instant, so cutting by time puts the cut at the same
    instant in every run.
"""

from dataclasses import dataclass

import numpy as np

from edmdc_tank_observables.data.loader import TS_NOMINAL, RawDataset

# Physical range of the tank level, in % of level.
LEVEL_RANGE = (0.0, 100.0)
# Physical range of the pump control signal, in % of actuation.
CONTROL_RANGE = (0.0, 100.0)


@dataclass
class PreprocessedDataset:
    """One run after preprocessing.

    The samples are consecutive in time and exactly ``ts`` apart, so every
    pair (k, k+1) is a real transition of the system. As returned by
    :func:`prepare`, h, u and sp are normalized to [0, 1].

    Attributes
    ----------
    h : numpy.ndarray, shape (T,)
        Tank level, normalized with h_scale.
    u : numpy.ndarray, shape (T,)
        Pump control signal, normalized with u_scale.
    sp : numpy.ndarray, shape (T,)
        Level setpoint of the control loop. It is a level, so it is
        normalized with h_scale. It is not used to identify the model; its
        steps tell where the system is in transient.
    t : numpy.ndarray, shape (T,)
        Time of each sample, in seconds from the start of the run. A part
        obtained from :func:`temporal_split` keeps the time of the full run.
    ts : float
        Sampling period, in seconds.
    h_scale : tuple of float
        Pair (h_min, h_max), in % of level, used to normalize h and sp.
    u_scale : tuple of float
        Pair (u_min, u_max), in % of actuation, used to normalize u.
    name : str
        Name of the run, e.g. "Baseline_50pct", with the suffix _train or
        _test if it comes from a split.
    """

    h: np.ndarray
    u: np.ndarray
    sp: np.ndarray
    t: np.ndarray
    ts: float
    h_scale: tuple[float, float]
    u_scale: tuple[float, float]
    name: str

    def __len__(self) -> int:
        return len(self.h)

    @property
    def n_pairs(self) -> int:
        """Pairs (k, k+1) usable for the regression."""
        return max(len(self.h) - 1, 0)

    def denormalize_h(self, h_norm: np.ndarray) -> np.ndarray:
        """Convert a signal in the normalized domain of h back to % of level.

        Parameters
        ----------
        h_norm : numpy.ndarray
            Signal in the same normalized domain as self.h.

        Returns
        -------
        numpy.ndarray
            The signal in % of level.
        """
        return denormalize(h_norm, self.h_scale)

    def denormalize_h_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Convert a difference in the normalized domain of h to % of level.

        Parameters
        ----------
        delta_norm : numpy.ndarray
            Difference (e.g. an error) in the same normalized domain as self.h.

        Returns
        -------
        numpy.ndarray
            The difference in % of level.
        """
        return denormalize_delta(delta_norm, self.h_scale)


def resample(raw: RawDataset, ts: float):
    """Decimate a run to period ts, keeping the real time of each sample.

    A sample is kept if its time t_s is a multiple of ts. Time is taken from
    the data and not from the row position, so rows removed as sensor faults
    do not shift the grid: a fault on a grid time leaves a gap, a fault off
    the grid has no effect.

    Parameters
    ----------
    raw : RawDataset
        Run returned by the loader.
    ts : float
        Sampling period, in seconds. It must be a positive whole multiple of
        TS_NOMINAL.

    Returns
    -------
    h, u, sp, t : numpy.ndarray
        Level (%), control (%), setpoint (%) and time (s) of the kept samples.

    Raises
    ------
    ValueError
        If ts is not a positive whole multiple of TS_NOMINAL.
    """
    factor = ts / TS_NOMINAL
    if factor < 1 or abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"ts={ts} s must be a positive whole multiple of {TS_NOMINAL} s.")

    t_all = raw.df["t_s"].to_numpy(dtype=float)

    remainder = np.mod(t_all, ts)
    on_grid = np.isclose(remainder, 0.0) | np.isclose(remainder, ts)
    idx = np.flatnonzero(on_grid)

    h = raw.df["Level_pct"].to_numpy(dtype=float)[idx]
    u = raw.df["Control_Signal_pct"].to_numpy(dtype=float)[idx]
    sp = raw.df["Setpoint_pct"].to_numpy(dtype=float)[idx]
    return h, u, sp, t_all[idx]


def find_gaps(t: np.ndarray, ts: float) -> np.ndarray:
    """Locate the pairs of consecutive samples that are not exactly ts apart.

    Parameters
    ----------
    t : numpy.ndarray
        Sample times, in seconds.
    ts : float
        Expected sampling period, in seconds.

    Returns
    -------
    numpy.ndarray of int
        Indices k such that the pair (k, k+1) crosses a gap. Empty if the
        series is contiguous.
    """
    if len(t) < 2:
        return np.empty(0, dtype=int)
    return np.flatnonzero(~np.isclose(np.diff(t), ts))


def normalize(x: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Normalize a signal to [0, 1] with the given range.

    Parameters
    ----------
    x : numpy.ndarray
        Signal in physical units.
    scale : tuple of float
        Pair (x_min, x_max) with the range of the variable.

    Returns
    -------
    numpy.ndarray
        The normalized signal, (x - x_min) / (x_max - x_min).

    Raises
    ------
    ValueError
        If the range is degenerate (x_max <= x_min).
    """
    x_min, x_max = scale
    if x_max <= x_min:
        raise ValueError(f"Degenerate scale: min={x_min}, max={x_max}.")
    return (x - x_min) / (x_max - x_min)


def denormalize(x_norm: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Undo :func:`normalize`.

    Parameters
    ----------
    x_norm : numpy.ndarray
        Signal in the normalized domain.
    scale : tuple of float
        Pair (x_min, x_max) used to normalize it.

    Returns
    -------
    numpy.ndarray
        The signal in the original units, x_norm * (x_max - x_min) + x_min.
    """
    x_min, x_max = scale
    return x_norm * (x_max - x_min) + x_min


def denormalize_delta(delta_norm: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Undo :func:`normalize` on a difference, such as an error.

    The minimum of the range is not added: a difference between two normalized
    values only has to be rescaled, and adding the offset would turn an error
    of 0 into a non-zero one.

    Parameters
    ----------
    delta_norm : numpy.ndarray
        Difference in the normalized domain.
    scale : tuple of float
        Pair (x_min, x_max) used to normalize the signals.

    Returns
    -------
    numpy.ndarray
        The difference in the original units, delta_norm * (x_max - x_min).
    """
    x_min, x_max = scale
    return delta_norm * (x_max - x_min)


def prepare(raw: RawDataset, ts: float) -> PreprocessedDataset:
    """Resample a run to ts, check it has no gaps and normalize it.

    Parameters
    ----------
    raw : RawDataset
        Run returned by the loader.
    ts : float
        Sampling period, in seconds.

    Returns
    -------
    PreprocessedDataset
        Contiguous series with h and u normalized with LEVEL_RANGE and
        CONTROL_RANGE, and sp normalized with LEVEL_RANGE.

    Raises
    ------
    ValueError
        If ts is not a positive whole multiple of TS_NOMINAL, or if any gap
        remains after resampling; the message gives the number of gaps and the
        times (s) after which the first ones occur.
    """
    h, u, sp, t = resample(raw, ts)
    gaps = find_gaps(t, ts)

    if len(gaps) > 0:
        first = ", ".join(f"{t[k]:g}" for k in gaps[:5])
        more = ", ..." if len(gaps) > 5 else ""
        raise ValueError(
            f"{raw.name} at Ts={ts:g} s: {len(gaps)} gap(s) over {len(t) - 1} pairs "
            f"(first after t = {first}{more} s)."
        )

    return PreprocessedDataset(
        h=normalize(h, LEVEL_RANGE),
        u=normalize(u, CONTROL_RANGE),
        sp=normalize(sp, LEVEL_RANGE),
        t=t,
        ts=float(ts),
        h_scale=LEVEL_RANGE,
        u_scale=CONTROL_RANGE,
        name=raw.name,
    )


def common_end_time(datasets) -> float:
    """Final instant shared by a set of runs.

    The temporal split measures its fraction from t = 0 to one sampling period
    after the last sample of each run (t_last + ts), so runs cut together must
    end at the same instant to be cut at the same one.

    Parameters
    ----------
    datasets : iterable of PreprocessedDataset
        Runs to be split together.

    Returns
    -------
    float
        Time of the last sample, in seconds, common to every run.

    Raises
    ------
    ValueError
        If there are no runs, or if their last samples are not at the same
        instant; the message lists each run with its end time.
    """
    ends = {d.name: float(d.t[-1]) for d in datasets}
    if not ends:
        raise ValueError("No runs given.")
    if len(set(ends.values())) > 1:
        listed = ", ".join(f"{name}: {t:g} s" for name, t in ends.items())
        raise ValueError(f"The runs do not end at the same instant, so they cannot be cut at the same one ({listed}).")
    return next(iter(ends.values()))


def temporal_split(dataset: PreprocessedDataset, train_frac: float = 0.7):
    """Split a run into fit and evaluation parts with a temporal cut at an absolute instant.

    The cut instant is train_frac x (t_last + ts), the given fraction of the
    acquisition time counted from t = 0, rounded to the nearest multiple of ts
    (a halfway case goes to the even multiple, as Python's round does) so that
    floating-point error cannot move it by one sample. The evaluation
    part starts at the first sample with t >= that instant. Runs that end at
    the same instant are therefore cut at the same instant, whatever their
    first sample (see :func:`common_end_time`). The pair that crosses the cut
    belongs to neither part, so no transition is shared.

    Parameters
    ----------
    dataset : PreprocessedDataset
        Complete run, as returned by :func:`prepare`, or a fit part to be
        split again.
    train_frac : float, default 0.7
        Fraction of the acquisition time that goes to the fit part.

    Returns
    -------
    train, test : PreprocessedDataset
        The two parts, named with the suffixes _train and _test. Both keep the
        absolute time t of the full run and its scales.

    Raises
    ------
    ValueError
        If train_frac is not in (0, 1), or if the cut leaves either part empty.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1).")

    n = len(dataset)
    ts = dataset.ts
    cut_time = round(train_frac * (dataset.t[-1] + ts) / ts) * ts
    # Every t is on the grid of multiples of ts, so half a period of tolerance
    # only absorbs floating-point error.
    cut = int(np.searchsorted(dataset.t, cut_time - 0.5 * ts, side="right"))
    if cut < 1 or cut >= n:
        raise ValueError(
            f"train_frac={train_frac} leaves one part empty (cut at t = {cut_time:g} s, "
            f"samples from {dataset.t[0]:g} to {dataset.t[-1]:g} s)."
        )

    def part(start, end, suffix):
        return PreprocessedDataset(
            h=dataset.h[start:end],
            u=dataset.u[start:end],
            sp=dataset.sp[start:end],
            t=dataset.t[start:end],
            ts=dataset.ts,
            h_scale=dataset.h_scale,
            u_scale=dataset.u_scale,
            name=dataset.name + suffix,
        )

    return part(0, cut, "_train"), part(cut, n, "_test")
