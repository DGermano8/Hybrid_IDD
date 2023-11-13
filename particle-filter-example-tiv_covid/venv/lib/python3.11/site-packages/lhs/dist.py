import importlib
import numpy as np
import scipy.stats
import sys


def sample_from(samples, dist_name, dist_kwargs):
    """
    Sample from a distribution by evaluating the quantile function.

    :param samples: The values for which to evaluate the quantile function.
    :param dist_name: The name of the distribution to sample.
    :param dist_kwargs: The (distribution-specific) shape parameters.
    :returns: The sample values as a ``numpy.ndarray`` that has the same shape
        as ``samples``.
    :raises ValueError: if the distribution ``dist_name`` is not defined.

    :Examples:

    >>> import lhs.dist
    >>> import numpy as np
    >>> samples = np.array([0.1, 0.5, 0.9])
    >>> kwargs = {'loc': 10, 'scale': 5}
    >>> values = lhs.dist.sample_from(samples, 'uniform', kwargs)
    >>> print(values)
    [10.5 12.5 14.5]
    """
    try:
        final_period = dist_name.rfind('.')
        if final_period < 0:
            # No period, look for a function in this module.
            fn = getattr(sys.modules[__name__], dist_name)
        else:
            # Also support fully-qualified function names.
            module_name = dist_name[:final_period]
            value_name = dist_name[final_period + 1:]
            module = importlib.import_module(module_name)
            fn = getattr(module, value_name)
    except (AttributeError, ModuleNotFoundError):
        return scipy_stats_dist(samples, dist_name, dist_kwargs)

    if not callable(fn):
        name = __name__ + '.' + dist_name
        raise ValueError('The value "{}" is not callable'.format(name))

    return fn(samples, **dist_kwargs)


def constant(samples, value):
    """
    The constant distribution, which always returns ``value``.

    :Examples:

    .. code-block:: python

       dist_R0 = {'name': 'constant', 'args': {'value': 2.53}}
    """
    return value * np.ones(samples.shape)


def inverse_uniform(samples, **kwargs):
    r"""
    The continuous inverse-uniform distribution, where
    :math:`X \sim \left[ \mathcal{U}(a, b) \right]^{-1}`.

    The lower and upper bounds may be defined in terms of the uniform
    distribution parameters (``low`` and ``high``), or in terms of their
    reciprocal (``inv_low`` and ``inv_high``).
    Any combination of these parameters may be used; see the examples below.

    :Examples:

    All four combinations of the uniform/reciprocal parameters produce
    identical results:

    >>> from lhs.dist import inverse_uniform
    >>> from lhs.sample import lhs_values
    >>> # Define the sample locations.
    >>> samples = [0, 0.5, 1]
    >>> param_name = 'alpha'
    >>> param_dist = {'name': 'inverse_uniform'}
    >>> # Define the distribution in terms of 'low' and 'high'.
    >>> param_dist['args'] = {'low': 5, 'high': 10}
    >>> lhs_values(param_name, param_dist, samples)
    array([0.1       , 0.13333333, 0.2       ])
    >>> # Define the distribution in terms of 'inv_low' and 'high'.
    >>> param_dist['args'] = {'inv_low': 0.2, 'high': 10}
    >>> lhs_values(param_name, param_dist, samples)
    array([0.1       , 0.13333333, 0.2       ])
    >>> # Define the distribution in terms of 'low' and 'inv_high'.
    >>> param_dist['args'] = {'low': 5, 'inv_high': 0.1}
    >>> lhs_values(param_name, param_dist, samples)
    array([0.1       , 0.13333333, 0.2       ])
    >>> # Define the distribution in terms of 'inv_low' and 'inv_high'.
    >>> param_dist['args'] = {'inv_low': 0.2, 'inv_high': 0.1}
    >>> lhs_values(param_name, param_dist, samples)
    array([0.1       , 0.13333333, 0.2       ])

    Samples are drawn between the values of the lower and upper bounds, even
    when the lower bounds are greater than the upper bounds:

    >>> import numpy as np
    >>> from lhs.dist import inverse_uniform
    >>> inverse_uniform(0.5, low=4, high=[6, 4, 2])
    array([0.2       , 0.25      , 0.33333333])
    """
    if len(kwargs) != 2:
        msg = 'inverse_uniform requires 2 arguments, but received {}'
        raise ValueError(msg.format(len(kwargs)))

    if 'low' in kwargs and 'inv_low' in kwargs:
        raise ValueError('inverse_uniform: cannot use low and inv_low')
    if 'high' in kwargs and 'inv_high' in kwargs:
        raise ValueError('inverse_uniform: cannot use high and inv_high')

    if 'low' in kwargs:
        low = kwargs['low']
    elif 'inv_low' in kwargs:
        low = 1 / kwargs['inv_low']
    else:
        raise ValueError('inverse_uniform: no low or inv_low argument')

    if 'high' in kwargs:
        high = kwargs['high']
    elif 'inv_high' in kwargs:
        high = 1 / kwargs['inv_high']
    else:
        raise ValueError('inverse_uniform: no high or inv_high argument')

    # Broadcast the two arrays to a common shape, and determine the
    # element-wise minimums and maximums.
    loc = np.minimum(low, high)
    scale = np.maximum(low, high) - loc

    # NOTE: we flip the samples so that the reciprocal values are ordered from
    # smallest to largest.
    dist = scipy.stats.uniform(loc=loc, scale=scale)
    values = dist.ppf(1 - np.array(samples))

    # The percent point function returns NaN when scale is zero, even though
    # random samples can be drawn from this zero-length interval.
    zero_scale = np.atleast_1d(scale == 0)
    if any(zero_scale):
        values[zero_scale] = np.broadcast_to(loc, values.shape)[zero_scale]

    reciprocals = 1.0 / values
    return reciprocals


def scipy_stats_dist(samples, dist_name, dist_kwargs):
    """
    Sample from a distribution defined in the ``scipy.stats`` module.

    :param samples: The values for which to evaluate the quantile function.
    :param dist_name: The name of the distribution to sample.
    :param dist_kwargs: The (distribution-specific) shape parameters.
    :returns: The sample values as a ``numpy.ndarray`` that has the same shape
        as ``samples``.
    :raises ValueError: if the distribution ``dist_name`` is not defined.
    """
    try:
        dist_class = getattr(scipy.stats, dist_name)
    except AttributeError:
        raise ValueError('Unknown distribution "{}"'.format(dist_name))
    dist_obj = dist_class(**dist_kwargs)
    return dist_obj.ppf(samples)
