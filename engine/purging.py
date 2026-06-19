from typing import Tuple


def purged_train_end(train_start: int, train_end: int, test_start: int,
                      label_horizon: int, embargo: int = 0) -> int:
    """Return an adjusted train_end that drops the trailing training samples
    whose forward-looking labels would overlap the test set.

    A label at index t depends on bars [t+1, t+label_horizon] (e.g. a triple
    barrier or fixed-horizon return). If t is in the training set but
    t+label_horizon falls into the test set, that label encodes information
    about test-period prices -- a subtle leak even though no feature crosses
    the boundary. The fix (Lopez de Prado: purging) is to drop training
    samples with t >= test_start - label_horizon - embargo.

    `embargo` adds an extra buffer beyond the label horizon for residual
    serial correlation; it defaults to 0 since label_horizon already covers
    the direct leakage path for this project's labeling schemes.

    Returns train_start unchanged if the entire training block would be
    purged (caller should treat train_start == returned value as "no usable
    training data").
    """
    if label_horizon < 0:
        raise ValueError(f"label_horizon must be >= 0, got {label_horizon}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    cutoff = test_start - label_horizon - embargo
    new_end = min(train_end, cutoff)
    return max(train_start, new_end)


def purge_window(train: Tuple[int, int], test: Tuple[int, int],
                  label_horizon: int, embargo: int = 0) -> Tuple[int, int]:
    """Convenience wrapper: given (train_start, train_end) and
    (test_start, test_end), return the purged (train_start, train_end)."""
    a, b = train
    c, _ = test
    return a, purged_train_end(a, b, c, label_horizon, embargo)


def n_purged(train: Tuple[int, int], test: Tuple[int, int],
              label_horizon: int, embargo: int = 0) -> int:
    a, b = train
    _, new_b = purge_window(train, test, label_horizon, embargo)
    return b - new_b
