"""Various methods for resampling particles."""

import logging
import numpy as np

from . import state


def post_regularise(ctx, px, new_px):
    """
    Sample model parameter values from a continuous approximation of the
    optimal filter, assuming that it has a smooth density.

    This is the post-regularised particle filter (post-RPF). For details, see
    chapter 12 of Doucet et al., Sequential Monte Carlo Methods in Practice,
    Springer, 2001.
    `doi:10.1007/978-1-4757-3437-9_12
    <https://doi.org/10.1007/978-1-4757-3437-9_12>`_

    :param ctx: The simulation context.
    :param px: The particles, prior to resampling.
    :param new_px: The particles after resampling directly from the discrete
        distribution (``px``). This matrix will be **destructively updated**
        with model parameter values samples from the regularisation kernel.
    """
    state.require_history_matrix(ctx, px)

    from . import stats

    logger = logging.getLogger(__name__)

    rnd = ctx.component['random']['resample']
    # NOTE: new_px may have fewer elements than px.
    count = new_px.shape[0]

    # Identify the parameters for which bounds have been provided.
    bounds = ctx.settings['filter']['regularisation']['bounds']

    # Only resample parameters that can be sampled continuously.
    model = ctx.component['model']
    smooth_fields = model.can_smooth()
    # List parameters in the order they appear in the state vector.
    smooth_fields = [n for n in bounds if n in smooth_fields]
    num_params = len(smooth_fields)

    if num_params == 0:
        logger.debug("Post-RPF: no parameters to resample")
        return

    # Copy the parameter columns into a new (contiguous) array.
    x = state.repack(px['state_vec'][smooth_fields])

    # Check for parameters that are constant (or nearly so) for all particles.
    # These parameters must be ignored or the covariance matrix will not be
    # positive definite, and the Cholesky decomposition will fail.
    p_range = np.ptp(x, axis=0)
    toln = ctx.settings['filter']['regularisation']['tolerance']
    good = p_range >= toln
    if not np.all(good):
        bad = np.logical_not(good)
        msg = "Post-RPF found {} constant parameter(s) at {}".format(
            sum(bad), np.array(smooth_fields)[bad])
        logger.debug(msg)

        # Update the variables related to these parameters.
        smooth_fields = [name for name in np.array(smooth_fields)[good]]
        num_params = len(smooth_fields)

        if num_params == 0:
            logger.debug("Post-RPF: no non-constant parameters to resample")
            return

        # Copy the parameter columns into a new (contiguous) array.
        x = state.repack(px['state_vec'][smooth_fields])

    # Note the fields that are being smoothed.
    logger.debug('Post-RPF: smoothing {}'.format(', '.join(smooth_fields)))

    # Use a bandwidth that is half that of the optimal bandwidth for a
    # Gaussian kernel (when the underlying density is Gaussian with unit
    # covariance), to handle multi-model densities.
    h = 0.5 * (4 / (count * (num_params + 2))) ** (1 / (num_params + 4))

    # Calculate the Cholesky decomposition of the parameter covariance
    # matrix V, which is used to transform independent normal samples
    # into multivariate normal samples with covariance matrix V.
    try:
        cov_mat = stats.cov_wt(x, px['weight'])
    except FloatingPointError as e:
        # NOTE: this can occur when essentially all of the probability mass is
        # associated with a single particle; for example, when the
        # second-biggest weight is on the order of 1e-20.
        logger.warning('Post-RPF: cannot calculate the covariance matrix')
        logger.warning(e)
        if ctx.settings['filter']['regularisation']['regularise_or_fail']:
            raise
        else:
            return
    try:
        a_mat = np.linalg.cholesky(cov_mat)
    except np.linalg.LinAlgError as e:
        # When the covariance matrix is not positive definite, print the name
        # and range of each parameter, and the covariance matrix itself.
        names = smooth_fields
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        means = np.mean(x, axis=0)
        mat_lines = str(cov_mat).splitlines()
        mat_sep = "\n      "
        mat_disp = mat_sep.join(["Covariance matrix:"] + mat_lines)
        logger = logging.getLogger(__name__)
        logger.warning("Post-RPF Cholesky decomposition: {}".format(e))
        logger.warning("Post-RPF parameters: {}".format(", ".join(names)))
        logger.warning("Minimum values: {}".format(mins))
        logger.warning("Maximum values: {}".format(maxs))
        logger.warning("Mean values:    {}".format(means))
        logger.warning(mat_disp)
        if ctx.settings['filter']['regularisation']['regularise_or_fail']:
            raise
        else:
            return

    # Sample the multivariate normal with covariance V and mean of zero.
    std_samples = rnd.normal(size=(num_params, count))
    scaled_samples = np.transpose(np.dot(a_mat, h * std_samples))

    # Add the sampled noise and clip to respect parameter bounds.
    for (ix, name) in enumerate(smooth_fields):
        min_val = bounds[name].get('min')
        max_val = bounds[name].get('max')
        new_px['state_vec'][name] += scaled_samples[:, ix]
        if min_val is not None or max_val is not None:
            # There are lower and/or upper bounds to apply.
            new_px['state_vec'][name] = np.clip(
                new_px['state_vec'][name],
                min_val, max_val)


def resample(ctx, px, net_weight=1.0, ix_offset=0, res_px=None, res_frac=0.0, res_offset=0):
    """Resample a particle population.

    :param ctx: The simulation context.
    :param px: An array of particle state vectors.
    :param net_weight: The total weight to allocate across the particles.
        Note that this includes the reservoir particles (if any).
    :param ix_offset: The index in the history matrix at which the particles
        in ``px`` begin.
    :param res_px: An optional array of reservoir particles.
    :param res_frac: The fraction of particles that should be selected from
        the reservoir particles.
    :param res_offset: The index in the history matrix at which the particles
        in ``res_px`` begin.

    The supported resampling methods are:

    - ``'basic'``:         uniform random numbers from [0, 1].
    - ``'stratified'``:    uniform random numbers from [j / m, (j + 1) / m).
    - ``'deterministic'``: select (j + a) / m for some fixed a in [0, 1].

    Where m is the number of particles and j = 0, ..., m - 1.

    These algorithms are described in G Kitagawa, J Comp Graph Stat
    5(1):1-25, 1996.
    `doi:10.2307/1390750 <https://doi.org/10.2307/1390750>`_

    When reservoir particles are provided, the following steps are performed:

    1. Sample C candidates from the N particles in ``px``;
    2. Apply post-regularisation, if enabled, to these C candidates; and
    3. Sample R candidates from ``res_px``.

    In this case, ``px`` contains ``N = C + R`` particles and ``R`` is equal
    to ``res_frac * N`` (rounded to the nearest integer).
    Note that post-regularisation, if enabled, is not applied to the reservoir
    particles.
    """
    logger = logging.getLogger(__name__)

    # Identify how many particles to select from each source.
    n_px = len(px)
    n_from_res = round(n_px * res_frac)
    n_from_px = n_px - n_from_res
    use_reservoir = res_px is not None and n_from_res > 0

    # Check that the particle array has the required fields.
    # NOTE: we manually check that each of these fields exists rather than
    # calling state.require_history_matrix(), because this allows us to write
    # test cases for resample() that can use a simple context scaffold.
    if px.dtype.names is None:
        raise ValueError('Cannot resample without required fields')
    for name in ['weight', 'prev_ix']:
        if name not in px.dtype.names:
            raise ValueError('Cannot resample without {} field'.format(name))
    # Resample the particles according to their weights.
    method = ctx.settings['filter']['resample']['method']
    rnd = ctx.component['random']['resample']
    sample_ixs = resample_ixs(px['weight'], rnd, method, count=n_from_px)
    num_unique = len(np.unique(sample_ixs))
    logger.debug('{} unique particles out of {}'.format(
        num_unique, len(sample_ixs)))
    # Construct the new particle array.
    new_px = np.copy(px[sample_ixs])
    # NOTE: apply the provided particle offset.
    new_px['prev_ix'] = sample_ixs + ix_offset
    new_px['weight'] = net_weight / n_px
    # Sample model parameter values from a regularised kernel, if requested.
    if ctx.settings['filter']['regularisation']['enabled']:
        ctx.call_event_handlers('before_regularisation', ctx, new_px)
        post_regularise(ctx, px, new_px)
        ctx.call_event_handlers('after_regularisation', ctx, new_px)
    # Copy the resampled particles back into the original array.
    px[:n_from_px] = new_px[:]
    # Sample particles from the reservoir, as required.
    if use_reservoir:
        res_weights = np.ones(len(res_px)) / len(res_px)
        res_sample_ixs = resample_ixs(res_weights, rnd, method,
                                      count=n_from_res)
        res_new_px = np.copy(res_px[res_sample_ixs])
        # NOTE: apply the provided reservoir particle offset.
        res_new_px['prev_ix'] = res_sample_ixs + res_offset
        res_new_px['weight'] = net_weight / n_px
        px[n_from_px:] = res_new_px[:]


def resample_weights(weights, rnd, method='deterministic', count=None):
    """
    Resample a particle weight array.

    :param np.ndarray weights: The particle weights.
    :param rnd: A random number generator.
    :param method: The resampling method: ``'basic'``, ``'stratified'``, or
        ``'deterministic'`` (default).
    :param count: The number of particles to select (default: the length of
        the ``weights`` array).
    :returns: A ``(sample_ixs, weight)`` tuple, where ``sample_ixs`` are the
        indices of the resampled particles and ``weight`` is the new weight
        for each particle (a single float).
    """
    sample_ixs = resample_ixs(weights, rnd, method, count=count)
    new_weight = 1 / len(sample_ixs)
    return (sample_ixs, new_weight)


def resample_ixs(weights, rnd, method, count=None):
    """
    Resample a particle weight array.

    :param np.ndarray weights: The particle weights.
    :param rnd: A random number generator.
    :param method: The resampling method: ``'basic'``, ``'stratified'``, or
        ``'deterministic'`` (default).
    :param count: The number of particles to select (default: the length of
        the ``weights`` array).
    :returns: An array that contains the index of each resampled particle.
    """
    if weights.ndim != 1:
        raise ValueError('weights array must be 1-dimensional')
    # Sort the particle indices according to weight (in descending order), so
    # that we can determine the original index of each resampled particle.
    # Use the merge sort algorithm because it is stable (thus preserving the
    # behaviour of Python's built-in `sorted` function).
    sorted_ix = np.argsort(- weights, kind='mergesort')
    # Sort the weights in descending order.
    sorted_ws = weights[sorted_ix]
    if count is None:
        count = len(weights)
    # Generate the random samples using the specified resampling method.
    if method == 'basic':
        choices = np.sort(rnd.uniform(size=count))
    elif method == 'stratified':
        choices = (rnd.uniform(size=count) + np.arange(count)) / count
    elif method == 'deterministic':
        choices = (rnd.uniform() + np.arange(count)) / count
    else:
        raise ValueError("Invalid resampling method '{}'".format(method))
    # Construct an array to record the index of each resampled particle.
    new_ixs = np.zeros(count, dtype=np.int_)
    # Calculate the upper bounds for each interval.
    # NOTE: normalise bounds to span the unit interval.
    bounds = np.cumsum(sorted_ws) / np.sum(sorted_ws)
    # Since the intervals and random samples are both monotonic increasing, we
    # only need step through the samples and record the current interval.
    bounds_ix = 0
    for (j, rand_val) in enumerate(choices):
        while bounds[bounds_ix] < rand_val:
            bounds_ix += 1
        new_ixs[j] = sorted_ix[bounds_ix]
    return new_ixs
