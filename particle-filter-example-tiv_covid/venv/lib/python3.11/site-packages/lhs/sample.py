import itertools
import logging
import numpy as np

from .dist import sample_from


def sample_subspace(rng, n, shape=None, broadcast=None):
    """
    Return samples from ``n`` equal subsets of the unit interval (or for many
    unit intervals).

    :param rng: A random number generator.
    :param n: The number of subsets to sample.
    :param shape: The (optional) number of unit intervals to sample.
        This can be an integer, or a sequence of integers (list or tuple).
    :param broadcast: An (optional) list of integers used to broadcast the
        samples to additional dimensions.
        Values greater than 0 indicate a new dimension, values less than or
        equal to zero indicate an existing dimension.
        See the code listings below for an example.

    :Examples:

    >>> import lhs.sample
    >>> import numpy as np
    >>> # Draw 10 samples from the unit interval.
    >>> rng = np.random.default_rng(seed=20201217)
    >>> n = 10
    >>> samples = lhs.sample.sample_subspace(rng, n)
    >>> # Ensure there is one sample in each of [0, 0.1], [0.1, 0.2], etc.
    >>> lower_bounds = np.linspace(0, 1 - 1/n, num=n)
    >>> upper_bounds = np.linspace(1/n, 1, num=n)
    >>> for (lower, upper) in zip(lower_bounds, upper_bounds):
    ...     in_interval = np.logical_and(samples >= lower, samples <= upper)
    ...     assert sum(in_interval) == 1

    Example of broadcasting samples to additional dimensions:

    >>> import lhs.sample
    >>> import numpy as np
    >>> alpha_dist = {
    ...     'name': 'beta',
    ...     'args': {'a': 1, 'b': 1},
    ...     'shape': 2,
    ...     # Broadcast from (10 x 2) to (10 x 3 x 2 x 4).
    ...     'broadcast': [3, 0, 4],
    ... }
    >>> rng = np.random.default_rng(12345)
    >>> num_samples = 10
    >>> samples = lhs.draw(rng, num_samples, {'alpha': alpha_dist})
    >>> assert samples['alpha'].shape == (10, 3, 2, 4)
    """
    is_scalar = shape is None or (isinstance(shape, int) and shape == 1)

    # Draw the samples.
    if is_scalar:
        samples = (rng.permutation(n) + rng.random(size=n)) / n
    else:
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, (list, tuple)):
            msg_fmt = 'Invalid shape type {}'
            raise ValueError(msg_fmt.format(type(shape)))

        samples_shape = tuple((n, *shape))
        samples = np.zeros(samples_shape)
        for ixs in itertools.product(*[range(n) for n in shape]):
            # Construct the index into the permutations array.
            index = (slice(None, None), *ixs)
            # Select a random sample ordering for this index.
            samples[index] = (rng.permutation(n) + rng.random(size=n)) / n

    # Broadcast the samples to the desired output shape (if any).
    if broadcast is not None:
        # Construct the indexing slice.
        index = [None if b > 0 else slice(None, None)
                 for b in broadcast]
        index.insert(0, slice(None, None))
        index = tuple(index)

        # Determine the output shape.
        shape_out = [n]
        shape_ix = 1
        for b in broadcast:
            if b > 0:
                shape_out.append(b)
            else:
                shape_out.append(samples.shape[shape_ix])
                shape_ix += 1
        shape_out = tuple(shape_out)

        # Broadcast the samples to the desired shape.
        samples = np.broadcast_to(samples[index], shape_out)

    return samples


def sample_subspaces(rng, n, param_shapes):
    """
    Return samples from ``n`` equal subsets of the unit interval (or for many
    unit intervals) for each parameter in ``params``.

    :param rng: A random number generator.
    :param n: The number of subsets to sample.
    :param param_shapes: A sequence of ``(name, shape)`` and/or
        ``(name, shape, broadcast)`` tuples.
    """
    param_shapes = [
        (shape[0], shape[1], None) if len(shape) == 2 else shape
        for shape in param_shapes
    ]
    return {
        name: sample_subspace(rng, n, shape, broadcast)
        for (name, shape, broadcast) in param_shapes
    }


def lhs_values(name, dist, samples):
    """
    Return values drawn from the inverse CDF of ``dist`` for each sample in
    the unit interval.

    :param name: The parameter name (used for error messages).
    :param dist: The sampling distribution details.
    :param samples: Samples from the unit interval.

    :Examples:

    >>> import lhs.sample
    >>> import numpy as np
    >>> import scipy.stats
    >>> # Define the sample locations.
    >>> samples = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    >>> # Identify the sampling distribution by name.
    >>> dist_1 = {
    ...     'name': 'beta',
    ...     'args': {'a': 2, 'b': 5},
    ... }
    >>> # Provide the sampling distribution object.
    >>> dist_2 = {
    ...     'distribution': scipy.stats.beta(a=2, b=5),
    ... }
    >>> # Identify the sampling distribution percent point function.
    >>> dist_3 = {
    ...     'ppf': scipy.stats.beta(a=2, b=5).ppf,
    ... }
    >>> # Ensure we obtain the same values from all three distributions.
    >>> values_1 = lhs.sample.lhs_values('dist_1', dist_1, samples)
    >>> values_2 = lhs.sample.lhs_values('dist_2', dist_2, samples)
    >>> values_3 = lhs.sample.lhs_values('dist_3', dist_3, samples)
    >>> assert np.allclose(values_1, values_2)
    >>> assert np.allclose(values_2, values_3)
    >>> assert np.allclose(values_1, values_3)
    """
    logger = logging.getLogger(__name__)

    if 'distribution' in dist:
        return dist['distribution'].ppf(samples)
    elif 'ppf' in dist:
        return dist['ppf'](samples)

    dist_name = dist.get('name')
    if dist_name is None:
        raise ValueError('Missing prior function for {}'.format(name))
    elif not isinstance(dist_name, str):
        raise ValueError('Invalid prior function for {}'.format(name))

    dist_kwargs = dist.get('args')
    if dist_kwargs is None:
        raise ValueError('Missing prior arguments for {}'.format(name))
    elif not isinstance(dist_kwargs, dict):
        raise ValueError('Invalid prior arguments for {}'.format(name))

    expected_keys = ['name', 'args', 'shape', 'broadcast']
    extra_keys = [k for k in dist if k not in expected_keys]
    if extra_keys:
        logger.warning('Extra prior keys for %s: %s', name, extra_keys)

    return sample_from(samples, dist_name, dist_kwargs)
