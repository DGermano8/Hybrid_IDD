"""A Latin hypercube sampler for scenario modelling."""

from . import sample


def draw(rng, n, params, dep_params=None, dep_fn=None, values=None):
    """
    Return samples from the provided parameter distributions.

    :param rng: A random number generator.
    :param n: The number of subsets to sample.
    :param params: The independent parameter distributions.
    :param dep_params: The (optional) dependent parameter details.
    :param dep_fn: The (optional) function that defines dependent parameter
        distributions, given the sample values for each independent parameter.
    :param values: An (optional) table of parameter values that have already
        been sampled.
        This can be useful when some parameters are dependent on parameters
        whose sample values are read from, e.g., external data files.

    :raise ValueError: if only one of ``dep_params`` and ``dep_fn`` is set to
        ``None``.

    .. note::

       The order in which the distributions are defined **matters**.
       Samples are drawn for each distribution in turn.
       If you want to ensure that certain parameters have reproducible samples
       when drawing values for multiple simulations, the parameter ordering
       must be consistent.
       This means that additional parameters that are only defined for certain
       simulations should be defined **after** all of the common parameters.
       See the example below for a demonstration.

    :Examples:

    >>> import lhs
    >>> import numpy as np
    >>> # Define X ~ U(0, 1).
    >>> dist_x = {'x': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}}}
    >>> # Define X ~ U(0, 1) and Y ~ U(0, 1).
    >>> dist_xy = {
    ...     'x': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}},
    ...     'y': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}},
    ... }
    >>> # Define Y ~ U(0, 1) and X ~ U(0, 1).
    >>> dist_yx = {
    ...     'y': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}},
    ...     'x': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}},
    ... }
    >>> n = 10
    >>> # Draw samples for X.
    >>> rand = np.random.default_rng(seed=12345)
    >>> samples_x = lhs.draw(rand, n, dist_x)
    >>> # Draw samples for X and Y; we should obtain identical samples for X.
    >>> rand = np.random.default_rng(seed=12345)
    >>> samples_xy = lhs.draw(rand, n, dist_xy)
    >>> assert np.array_equal(samples_x['x'], samples_xy['x'])
    >>> # Draw samples for Y and X; we should obtain different samples for X.
    >>> rand = np.random.default_rng(seed=12345)
    >>> samples_yx = lhs.draw(rand, n, dist_yx)
    >>> assert not np.array_equal(samples_x['x'], samples_yx['x'])
    """

    if (dep_params is None) != (dep_fn is None):
        raise ValueError('Cannot provide only one of dep_params and dep_fn')

    if dep_params is None:
        dep_params = {}

    # Collect the independent and dependent parameter names.
    indep_names = list(params.keys())
    dep_names = list(dep_params.keys())

    # Collect the details for each parameter in turn.
    param_shapes = []
    for name in indep_names:
        param_shapes.append((name, params[name].get('shape'),
                             params[name].get('broadcast')))
    for name in dep_names:
        param_shapes.append((name, dep_params[name].get('shape'),
                             dep_params[name].get('broadcast')))

    # Draw subspace samples for each parameter.
    subspace_samples = sample.sample_subspaces(rng, n, param_shapes)

    # Obtain sample values for each independent parameter.
    if values is None:
        values = {}

    for name in indep_names:
        dist = params[name]
        values[name] = sample.lhs_values(name, dist, subspace_samples[name])

    # Obtain sample values for each dependent parameter.
    if dep_fn is not None:
        # Construct the dependent parameter distributions.
        dep_dists = dep_fn(values, dep_params)

        # Ensure that the expected distributions have been defined.
        expect_dists = set(dep_params.keys())
        actual_dists = set(dep_dists.keys())
        if expect_dists != actual_dists:
            raise ValueError('Inconsistent dependent parameters')

        # Obtain sample values for each dependent parameter in turn.
        for (name, dist) in dep_dists.items():
            values[name] = sample.lhs_values(name, dist,
                                             subspace_samples[name])

    return values
