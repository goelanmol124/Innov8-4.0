"""
Production oracle used during evaluation.

Participants never see this file. The oracle wraps the hidden evaluation
labels and enforces the query budget strictly. Labels are stored inside
the object and are inaccessible without calling __call__.
"""

import numpy as np


class BudgetExceededError(Exception):
    """Raised when a submission tries to query more indices than the budget allows."""


class Oracle:
    """
    Callable oracle that returns ground-truth labels for requested indices.

    Parameters
    ----------
    labels : array-like of int, shape (n,)
        Ground-truth labels for the evaluation dataset. Never exposed directly.
    budget : int
        Maximum total number of indices that may be queried across ALL calls.

    Usage
    -----
        oracle = Oracle(labels, budget=100)
        labels_for_5_rows = oracle([0, 3, 17, 42, 999])   # costs 5 budget
        labels_for_1_row  = oracle([7])                    # costs 1 budget

    Raises
    ------
    BudgetExceededError
        If the requested batch would push total queries past the budget.
    IndexError
        If any requested index is out of range.
    TypeError
        If indices is not iterable or contains non-integers.
    """

    def __init__(self, labels, budget: int = 100):
        self._labels = np.asarray(labels, dtype=int)
        self._budget = int(budget)
        self._used = 0

    def __call__(self, indices) -> list[int]:
        """
        Query labels for the given indices.

        Parameters
        ----------
        indices : list[int] or array-like
            Row indices into the dataset. May be any length, but the TOTAL
            number of indices across all calls must not exceed the budget.

        Returns
        -------
        list[int]  — labels (0 or 1) in the same order as indices.
        """
        try:
            indices = list(indices)
        except TypeError:
            raise TypeError("indices must be iterable (e.g. a list of ints)")

        if not indices:
            return []

        n_requested = len(indices)

        if self._used + n_requested > self._budget:
            remaining = self._budget - self._used
            raise BudgetExceededError(
                f"Budget exceeded. "
                f"Budget: {self._budget} | "
                f"Used so far: {self._used} | "
                f"Requested now: {n_requested} | "
                f"Remaining: {remaining}. "
                f"Split your query or request fewer indices."
            )

        n_rows = len(self._labels)
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(f"Index must be an integer, got {type(idx).__name__}: {idx!r}")
            if not (0 <= int(idx) < n_rows):
                raise IndexError(f"Index {idx} out of range [0, {n_rows})")

        self._used += n_requested
        return [int(self._labels[int(i)]) for i in indices]

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def queries_used(self) -> int:
        """Total indices queried so far."""
        return self._used

    @property
    def budget(self) -> int:
        """Maximum total indices allowed."""
        return self._budget

    @property
    def budget_remaining(self) -> int:
        """How many more indices may be queried."""
        return self._budget - self._used

    def __repr__(self) -> str:
        return (
            f"Oracle(budget={self._budget}, used={self._used}, "
            f"remaining={self.budget_remaining})"
        )
