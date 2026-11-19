"""EDMDc identification.

The identified model is

    Psi(h_{k+1}) ~ A @ Psi(h_k) + B @ u_k + c

where Psi is the observable dictionary, u the control signal and c a constant
term. Every dictionary has the level in its first component, so the predicted
level is the first component of the lifted state and no separate output matrix
is fitted.
"""

from dataclasses import dataclass

import numpy as np

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.observables.dictionaries import Dictionary


@dataclass
class EDMDcModel:
    """An identified EDMDc model.

    Attributes
    ----------
    A : numpy.ndarray, shape (p, p)
        Evolution matrix in the space of observables.
    B : numpy.ndarray, shape (p, 1)
        Input matrix.
    c : numpy.ndarray, shape (p,)
        Constant term.
    dictionary : Dictionary
        Dictionary of observables the model was fitted with.
    ts : float, default nan
        Sampling period of the fitting data, in seconds.
    n_pairs : int, default 0
        Pairs of consecutive samples used in the fit.
    cond : float, default nan
        Condition number of the regression matrix.
    fit_rmse_pct : float, default nan
        In-sample one-step RMSE of the level, in % of level.
    """

    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    dictionary: Dictionary
    ts: float = float("nan")
    n_pairs: int = 0
    cond: float = float("nan")
    fit_rmse_pct: float = float("nan")

    @property
    def p(self) -> int:
        """Dimension of the space of observables."""
        return self.A.shape[0]

    @property
    def spectral_radius(self) -> float:
        """Largest modulus among the eigenvalues of A.

        It is a property of the discrete-time matrix A in the lifted space and
        has no physical interpretation by itself.
        """
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def predict_step(self, psi: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance the lifted state one step.

        Parameters
        ----------
        psi : numpy.ndarray, shape (N, p)
            Lifted states at the current instant.
        u : numpy.ndarray, shape (N, 1)
            Control at the same instants, normalized like the fitting data.

        Returns
        -------
        numpy.ndarray, shape (N, p)
            Lifted states at the next instant.
        """
        psi = np.atleast_2d(psi)
        u = np.atleast_2d(u)
        return psi @ self.A.T + u @ self.B.T + self.c

    def simulate(self, h_history: np.ndarray, u_future: np.ndarray) -> np.ndarray:
        """Simulate the model in open loop over a horizon.

        u_future[0] is the control at the instant of the last value of
        h_history and u_future[i] the control i steps later; output i is the
        predicted level one step after the instant where u_future[i] is
        applied.
        The full lifted state is propagated linearly with A, B and c: delayed
        levels and sqrt(h) are not recomputed from the predicted level.

        Parameters
        ----------
        h_history : numpy.ndarray
            Level history used to lift the initial state, normalized like the
            fitting data. n_delays + 1 consecutive values are needed (one if
            the dictionary has no delays); if more are given, the last ones are
            used.
        u_future : numpy.ndarray, shape (H,)
            Control applied at each of the H steps of the horizon.

        Returns
        -------
        numpy.ndarray, shape (H,)
            Predicted level at each step, in the normalized domain of the data.

        Raises
        ------
        ValueError
            If h_history has fewer than n_delays + 1 values, or, from
            :meth:`Dictionary.transform`, if the dictionary is physical and
            the history used has a negative value.
        """
        dictionary = self.dictionary

        h_history = np.atleast_1d(np.asarray(h_history, dtype=float))
        needed = dictionary.n_delays + 1
        if len(h_history) < needed:
            raise ValueError(
                f"{dictionary.id}: {needed} samples of initial history are needed, "
                f"got {len(h_history)}."
            )
        u_future = np.asarray(u_future, dtype=float).ravel()

        # The last row of transform() is the lifted state at the most recent
        # instant of the history, which is where the horizon starts.
        psi = dictionary.transform(h_history[-needed:])[-1]

        prediction = np.empty(len(u_future))
        for k, u_k in enumerate(u_future):
            psi = self.A @ psi + self.B @ np.array([u_k]) + self.c
            prediction[k] = psi[0]

        return prediction


def fit_edmdc(dictionary: Dictionary, dataset: PreprocessedDataset) -> EDMDcModel:
    """Fit an EDMDc model on a preprocessed run.

    Builds the pairs (Psi_k, u_k) -> Psi_{k+1} and solves, by least squares,

        min over A, B, c of  sum_k || Psi_{k+1} - A Psi_k - B u_k - c ||^2

    Parameters
    ----------
    dictionary : Dictionary
        Dictionary of observables.
    dataset : PreprocessedDataset
        Run to fit on, with level and control normalized.

    Returns
    -------
    EDMDcModel
        The identified model, with the in-sample one-step RMSE of the level in
        % of level (using the level scale of the dataset).

    Raises
    ------
    ValueError
        If the series gives fewer pairs of consecutive samples than the
        regression has columns (p + 2).
    ValueError
        Propagated from :meth:`Dictionary.transform` if the series of the
        dataset has no more samples than n_delays, or if it has a negative
        level and the dictionary is physical.
    """
    psi = dictionary.transform(dataset.h)        # (T - n_delays, p)
    u = dictionary.align_input(dataset.u)        # (T - n_delays, 1)

    # Pairs of consecutive instants: lifted state and input at k, against the
    # lifted state at k + 1.
    psi_k, psi_k1, u_k = psi[:-1], psi[1:], u[:-1]

    # Regression matrix Z = [Psi_k, u_k, 1]. Psi_{k+1} = Z @ Theta is solved for
    # every component of Psi at once.
    z = np.column_stack([psi_k, u_k, np.ones((len(psi_k), 1))])
    if len(z) < z.shape[1]:
        raise ValueError(
            f"{dictionary.id} on {dataset.name}: {len(z)} pairs of consecutive samples "
            f"for {z.shape[1]} regressors; at least {z.shape[1]} are needed."
        )

    theta, *_ = np.linalg.lstsq(z, psi_k1, rcond=None)   # (p + 2, p)

    p = psi.shape[1]
    a = theta[:p].T
    b = theta[p : p + 1].T
    c = theta[p + 1]

    # One-step error of the level, in %.
    h_error = psi_k1[:, 0] - (z @ theta)[:, 0]
    h_error = dataset.denormalize_h_delta(h_error)

    return EDMDcModel(
        A=a,
        B=b,
        c=np.asarray(c, dtype=float).ravel(),
        dictionary=dictionary,
        ts=dataset.ts,
        n_pairs=len(psi_k),
        cond=float(np.linalg.cond(z)),
        fit_rmse_pct=float(np.sqrt(np.mean(h_error**2))),
    )
