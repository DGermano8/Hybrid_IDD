"""Weighted quantiles, credible intervals, and other statistics."""

import numpy as np


def cov_wt(x, wt, cor=False):
    r"""Estimate the weighted covariance or correlation matrix.

    Equivalent to ``cov.wt(x, wt, cor, center=TRUE, method="unbiased")`` as
    provided by the ``stats`` package for R.

    :param x: A 2-D array; columns represent variables and rows represent
        observations.
    :param wt: A 1-D array of observation weights.
    :param cor: Whether to return a correlation matrix instead of a covariance
        matrix.

    :return: The covariance matrix (if ``cor=False``) or the correlation
        matrix (if ``cor=True``).
    """

    if x.ndim != 2:
        raise ValueError("x has dimension {} != 2".format(x.ndim))

    if wt.ndim != 1:
        raise ValueError("weights have dimension {} != 1".format(wt.ndim))

    if wt.shape[0] != x.shape[0]:
        raise ValueError("{} observations but {} weights".format(x.shape[0],
                                                                 wt.shape[0]))

    if any(wt < 0):
        raise ValueError("negative weight(s) found")

    with np.errstate(divide='raise'):
        cov = np.cov(x.T, aweights=wt)

    if cor:
        # Convert the covariance matrix into a correlation matrix.
        sd = np.array([np.sqrt(np.diag(cov))])
        sd_t = np.transpose(sd)
        return np.atleast_2d(cov / sd / sd_t)
    else:
        return np.atleast_2d(cov)


def avg_var_wt(x, weights, biased=True):
    """
    Return the weighted average and variance (based on a Stack Overflow
    `answer <http://stackoverflow.com/a/2415343>`_).

    :param x: A 1-D array of values.
    :param weights: A 1-D array of **normalised** weights.
    :param biased: Use a biased variance estimator.

    :return: A tuple that contains the weighted average and weighted variance.

    :raises ValueError: if ``x`` or ``weights`` are not one-dimensional, or if
        ``x`` and ``weights`` have different dimensions.
    """
    if len(x.shape) != 1:
        message = 'x is not 1D and has shape {}'
        raise ValueError(message.format(x.shape))
    if len(weights.shape) != 1:
        message = 'weights is not 1D and has shape {}'
        raise ValueError(message.format(weights.shape))
    if x.shape != weights.shape:
        message = 'x and weights have different shapes {} and {}'
        raise ValueError(message.format(x.shape, weights.shape))

    average = np.average(x, weights=weights)
    # Fast and numerically precise biased estimator.
    variance = np.average((x - average) ** 2, weights=weights)
    if not biased:
        # Use an unbiased estimator for the population variance.
        variance /= (1 - np.sum(weights ** 2))
    return (average, variance)


def qtl_wt(x, weights, probs):
    """
    Calculate weighted quantiles of an array of values, where each value has a
    fractional weighting.

    Weights are summed over exact ties, yielding distinct values x_1 < x_2 <
    ... < x_N, with corresponding weights w_1, w_2, ..., w_N.
    Let ``s_j`` denote the sum of the first j weights, and let ``W`` denote
    the sum of all the weights.
    For a probability ``p``:

    - If ``p * W < s_1`` the estimated quantile is ``x_1``.
    - If ``s_j < p * W < s_{j + 1}`` the estimated quantile is ``x_{j + 1}``.
    - If ``p * W == s_N`` the estimated quantile is ``x_N``.
    - If ``p * W == s_j`` the estimated quantile is ``(x_j + x_{j + 1}) / 2``.

    :param x: A 1-D array of values.
    :param weights: A 1-D array of weights.
    :param probs: The quantile(s) to compute.

    :return: The array of weighted quantiles.

    :raises ValueError: if ``x`` or ``weights`` are not one-dimensional, or if
        ``x`` and ``weights`` have different dimensions.
    """

    if len(x.shape) != 1:
        message = 'x is not 1D and has shape {}'
        raise ValueError(message.format(x.shape))
    if len(weights.shape) != 1:
        message = 'weights is not 1D and has shape {}'
        raise ValueError(message.format(weights.shape))
    if x.shape != weights.shape:
        message = 'x and weights have different shapes {} and {}'
        raise ValueError(message.format(x.shape, weights.shape))

    # Remove values with zero or negative weights.
    # Weights of zero can arise if a particle is deemed sufficiently unlikely
    # given the recent observations and resampling has not (yet) occurred.
    if any(weights <= 0):
        mask = weights > 0
        weights = weights[mask]
        x = x[mask]

    # Sort x and the weights.
    i = np.argsort(x)
    x = x[i]
    weights = weights[i]

    # Combine duplicated values into a single sample.
    if any(np.diff(x) == 0):
        unique_xs = np.unique(x)
        weights = [sum(weights[v == x]) for v in unique_xs]
        x = unique_xs

    nx = len(x)
    cum_weights = np.cumsum(weights)
    net_weight = np.sum(weights)
    eval_cdf_locns = np.array(probs) * net_weight

    # Decide how strictly to compare probabilities to cumulative weights.
    atol = 1e-10

    # Define the bisection of lower and upper indices.
    def bisect(ix_lower, ix_upper):
        """
        Bisect the interval spanned by lower and upper indices.
        Returns ``None`` when the interval cannot be further divided.
        """
        if ix_upper > ix_lower + 1:
            return np.rint((ix_lower + ix_upper) / 2).astype(int)
        else:
            return None

    # Evaluate each quantile in turn.
    quantiles = np.zeros(len(probs))
    for (locn_ix, locn) in enumerate(eval_cdf_locns):
        # Check whether the quantile is the very first or last value.
        if cum_weights[0] >= (locn - atol):
            # NOTE: use strict equality with an absolute tolerance.
            if np.abs(locn - cum_weights[0]) <= atol and nx > 1:
                # Average over the two matching values.
                quantiles[locn_ix] = 0.5 * (x[0] + x[1])
            else:
                quantiles[locn_ix] = x[0]
            continue
        if cum_weights[-1] <= (locn - atol):
            quantiles[locn_ix] = x[-1]
            continue

        # Search the entire range of values.
        ix_lower = 0
        ix_upper = nx - 1

        # Find the smallest index in cum_weights that is greater than or equal
        # to the location at which to evaluate the CDF.
        ix_mid = bisect(ix_lower, ix_upper)
        while ix_mid is not None:
            w_mid = cum_weights[ix_mid]

            # NOTE: use strict equality with an absolute tolerance.
            if w_mid >= (locn - atol):
                ix_upper = ix_mid
            else:
                ix_lower = ix_mid
            ix_mid = bisect(ix_lower, ix_upper)

        # NOTE: use strict equality with an absolute tolerance.
        if np.abs(locn - cum_weights[ix_upper]) <= atol and ix_upper < nx - 1:
            # Average over the two matching values.
            quantiles[locn_ix] = 0.5 * (x[ix_upper] + x[ix_upper + 1])
        else:
            quantiles[locn_ix] = x[ix_upper]

    return quantiles


def cred_wt(x, weights, creds):
    """Calculate weighted credible intervals.

    :param x: A 1-D array of values.
    :param weights: A 1-D array of weights.
    :param creds: The credible interval(s) to compute (``0..100``, where ``0``
        represents the median and ``100`` the entire range).
    :type creds: List(int)

    :return: A dictionary that maps credible intervals to the lower and upper
        interval bounds.

    :raises ValueError: if ``x`` or ``weights`` are not one-dimensional, or if
        ``x`` and ``weights`` have different dimensions.
    """
    if len(x.shape) != 1:
        message = 'x is not 1D and has shape {}'
        raise ValueError(message.format(x.shape))
    if len(weights.shape) != 1:
        message = 'weights is not 1D and has shape {}'
        raise ValueError(message.format(weights.shape))
    if x.shape != weights.shape:
        message = 'x and weights have different shapes {} and {}'
        raise ValueError(message.format(x.shape, weights.shape))

    creds = sorted(creds)
    median = creds[0] == 0
    if median:
        creds = creds[1:]
    probs = [[0.5 - cred / 200.0, 0.5 + cred / 200.0] for cred in creds]
    probs = [pr for pr_list in probs for pr in pr_list]
    if median:
        probs = [0.5] + probs
    qtls = qtl_wt(x, weights, probs)
    intervals = {}
    if median:
        intervals[0] = (qtls[0], qtls[0])
        qtls = qtls[1:]
    for cred in creds:
        intervals[cred] = (qtls[0], qtls[1])
        qtls = qtls[2:]
    return intervals
