"""Observable dictionaries for EDMDc on the tank level.

A dictionary maps the measured level h to a vector of observables in which the
dynamics are approximated as linear. The eight configurations are:

    D0    state only                     [h]
    D1a   monomials                      [h, h^2, ..., h^order]
    D1b   Gaussian RBF                   [h, phi_1(h), ..., phi_M(h)]
    D1c   time delays                    [h_k, h_{k-1}, ..., h_{k-n}]
    D2    physical                       [h, sqrt(h)]
    D3a   monomials + physical           [h, h^2, ..., h^order, sqrt(h)]
    D3b   RBF + physical                 [h, phi_1(h), ..., phi_M(h), sqrt(h)]
    D3c   delays + physical              [h_k, h_{k-1}, ..., h_{k-n}, sqrt(h_k)]

The first column is always the level itself and the square root, when present,
is always the last one.
"""

from dataclasses import dataclass

import numpy as np

# Hyperparameters each configuration needs (_REQUIRED) and the configurations
# that add sqrt(h) (_PHYSICAL).
_REQUIRED = {
    "D0": (),
    "D1a": ("order",),
    "D1b": ("n_centers",),
    "D1c": ("n_delays",),
    "D2": (),
    "D3a": ("order",),
    "D3b": ("n_centers",),
    "D3c": ("n_delays",),
}
_PHYSICAL = {"D2", "D3a", "D3b", "D3c"}


@dataclass(frozen=True)
class Dictionary:
    """An observable dictionary: which observables to compute, and how many.

    Build the eight configurations with :func:`make_dictionary`.

    Attributes
    ----------
    id : str
        Short identifier of the configuration, e.g. "D1a".
    order : int or None
        Highest power of the monomials, or None if the dictionary has none.
    n_centers : int or None
        Number of Gaussian RBF centres, or None if the dictionary has none.
    n_delays : int
        Number of delayed levels, which is also the number of initial samples
        the dictionary consumes. 0 if the dictionary has no delays.
    physical : bool
        Whether the square root of the level is appended as the last column.
    rbf_range : tuple of float, default (0.0, 1.0)
        Interval in which the RBF centres are equally spaced. The default is
        the range of the normalized level.
    rbf_width : float or None, default None
        Width of the Gaussian kernel. None means the spacing between centres,
        or the width of rbf_range if there is a single centre.
    """

    id: str
    order: int | None
    n_centers: int | None
    n_delays: int
    physical: bool
    rbf_range: tuple[float, float] = (0.0, 1.0)
    rbf_width: float | None = None

    @property
    def dimension(self) -> int:
        """Number of observables p, which is the number of columns of transform."""
        p = 1
        if self.order is not None:
            p += self.order - 1
        if self.n_centers is not None:
            p += self.n_centers
        p += self.n_delays
        if self.physical:
            p += 1
        return p

    @property
    def observable_names(self) -> list[str]:
        """Name of each observable, one per column of transform and in the same order.

        Returns
        -------
        list of str
            "h", then "h^2".."h^order", then "rbf_1".."rbf_M", then
            "h_lag_1".."h_lag_n" and finally "sqrt_h", keeping only the
            groups the dictionary has. Its length is dimension.
        """
        names = ["h"]
        if self.order is not None:
            names += [f"h^{k}" for k in range(2, self.order + 1)]
        if self.n_centers is not None:
            names += [f"rbf_{j}" for j in range(1, self.n_centers + 1)]
        names += [f"h_lag_{j}" for j in range(1, self.n_delays + 1)]
        if self.physical:
            names.append("sqrt_h")
        return names

    def transform(self, h: np.ndarray) -> np.ndarray:
        """Map a level series to the observables.

        Parameters
        ----------
        h : numpy.ndarray, shape (T,)
            Normalized level series.

        Returns
        -------
        numpy.ndarray, shape (T - n_delays, dimension)
            Row i corresponds to sample i + n_delays of h. The columns are, in
            this order: h, the monomials h^2..h^order, the RBF columns, the
            delayed levels h_{k-1}..h_{k-n} and sqrt(h).

        Raises
        ------
        ValueError
            If the series has no more samples than n_delays, or if the
            dictionary is physical and any value of h is negative.
        """
        h = np.asarray(h, dtype=float).ravel()
        n = self.n_delays
        if len(h) <= n:
            raise ValueError(f"{self.id}: the series has {len(h)} samples, not enough for {n} delays.")
        if self.physical and np.any(h < 0):
            raise ValueError(f"{self.id}: the level has negative values, so its square root is undefined.")

        state = h[n:]
        columns = [state]

        if self.order is not None:
            columns += [state**k for k in range(2, self.order + 1)]

        if self.n_centers is not None:
            centers = np.linspace(self.rbf_range[0], self.rbf_range[1], self.n_centers)
            width = self.rbf_width
            if width is None:
                width = centers[1] - centers[0] if self.n_centers > 1 else self.rbf_range[1] - self.rbf_range[0]
            columns += [np.exp(-((state - c) ** 2) / (2.0 * width**2)) for c in centers]

        # h[n - j : T - j] is the series delayed j steps, aligned with state.
        columns += [h[n - j : len(h) - j] for j in range(1, n + 1)]

        if self.physical:
            columns.append(np.sqrt(state))

        return np.column_stack(columns)

    def align_input(self, u: np.ndarray) -> np.ndarray:
        """Trim the control so that row i of the observables and of the input are the same instant.

        Parameters
        ----------
        u : numpy.ndarray, shape (T,)
            Normalized control signal.

        Returns
        -------
        numpy.ndarray, shape (T - n_delays, 1)
            The control from sample n_delays on, as a column.
        """
        u = np.asarray(u, dtype=float).ravel()
        return u[self.n_delays :].reshape(-1, 1)


def make_dictionary(
    id: str,
    *,
    order: int | None = None,
    n_centers: int | None = None,
    n_delays: int | None = None,
) -> Dictionary:
    """Build one of the eight dictionary configurations.

    Each configuration needs exactly the hyperparameters of its family:
    order for D1a and D3a, n_centers for D1b and D3b, n_delays for D1c and
    D3c, and none for D0 and D2. The RBF range and width keep their defaults.

    Parameters
    ----------
    id : str
        One of "D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c".
    order : int, optional
        Highest power of the monomials. Must be at least 2.
    n_centers : int, optional
        Number of RBF centres. Must be at least 1.
    n_delays : int, optional
        Number of delayed levels. Must be at least 1.

    Returns
    -------
    Dictionary
        The configured dictionary.

    Raises
    ------
    ValueError
        If id is unknown, if a hyperparameter the configuration needs is
        missing, if one it does not use is given, or if order < 2,
        n_centers < 1 or n_delays < 1.
    """
    if id not in _REQUIRED:
        raise ValueError(f"Unknown dictionary {id!r}. Options: {sorted(_REQUIRED)}.")

    given = {"order": order, "n_centers": n_centers, "n_delays": n_delays}
    needed = _REQUIRED[id]
    missing = [name for name in needed if given[name] is None]
    unused = [name for name, value in given.items() if value is not None and name not in needed]
    if missing:
        raise ValueError(f"{id} requires {missing}.")
    if unused:
        raise ValueError(f"{id} does not use {unused}.")

    if order is not None and order < 2:
        raise ValueError("order must be >= 2 (order 1 is D0).")
    if n_centers is not None and n_centers < 1:
        raise ValueError("n_centers must be >= 1.")
    if n_delays is not None and n_delays < 1:
        raise ValueError("n_delays must be >= 1.")

    return Dictionary(
        id=id,
        order=order,
        n_centers=n_centers,
        n_delays=0 if n_delays is None else int(n_delays),
        physical=id in _PHYSICAL,
    )
