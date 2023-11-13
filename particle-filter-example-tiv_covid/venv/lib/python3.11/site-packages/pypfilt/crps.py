"""Calculate CRPS scores for simulated observations."""

import numpy as np


def crps_sample(true_values, samples_table):
    """
    Calculate the CRPS score for a table of samples drom from predictive
    distributions for multiple values, using the empirical distribution
    function defined by the provided samples.

    :param true_values: A 1-D array of observed values.
    :param samples_table: A 2-D array of samples, where each row contains the
        samples for the corresponding value in ``true_values``.
    """
    if np.ndim(true_values) != 1:
        raise ValueError('true_values must be a 1-D array')
    if np.ndim(samples_table) != 2:
        raise ValueError('samples_table must be a 2-D array')
    if len(true_values) != samples_table.shape[0]:
        raise ValueError('incompatible dimensions')
    return np.fromiter(
        (crps_edf_scalar(truth, samples_table[ix])
         for (ix, truth) in enumerate(true_values)),
        dtype=np.float_)


def crps_edf_scalar(true_value, samples):
    """
    Calculate the CRPS score for samples drawn from a predictive distribution
    for a single value, using the empirical distribution function defined by
    the provided samples.

    :param true_value: The (scalar) value that was observed.
    :param samples: Samples from the predictive distribution (a 1-D array).
    """
    c_1n = 1 / len(samples)
    x = np.sort(samples)
    a = np.arange(0.5 * c_1n, 1, c_1n)
    return 2 * c_1n * np.sum(((true_value < x) - a) * (x - true_value))


def simulated_obs_crps(true_obs, sim_obs):
    """
    Calculate CRPS scores for simulated observations, such as those recorded
    by the :class:`~pypfilt.summary.SimulatedObs` table, against observed
    values.

    The returned array has fields: ``'time'``, ``'fs_time'``, and ``'score'``.

    :param true_obs: The table of recorded observations; this must contain the
        fields ``'time'`` and ``'value``'.
    :param sim_obs: The table of simulated observations; this must contain the
        fields ``'fs_time'``, ``'time'``, and ``'value'``.

    :raises ValueError: if ``true_obs`` or ``sim_obs`` do not contain all of
        the required fields.
    """
    # Check that required columns are present.
    for column in ['time', 'value']:
        if column not in true_obs.dtype.names:
            msg_fmt = 'Column "{}" not found in true_obs'
            raise ValueError(msg_fmt.format(column))
    for column in ['fs_time', 'time', 'value']:
        if column not in sim_obs.dtype.names:
            msg_fmt = 'Column "{}" not found in sim_obs'
            raise ValueError(msg_fmt.format(column))

    # Only retain simulated observations for times with true observations.
    sim_mask = np.isin(sim_obs['time'], true_obs['time'])
    sim_obs = sim_obs[sim_mask]
    # Only retain true observations for times with simulated observations.
    true_mask = np.isin(true_obs['time'], sim_obs['time'])
    true_obs = true_obs[true_mask]

    # Identify the output rows.
    time_combs = np.unique(sim_obs[['fs_time', 'time']])
    score_rows = len(time_combs)
    time_dtype = true_obs.dtype.fields['time'][0]
    scores = np.zeros(
        (score_rows,),
        dtype=[('time', time_dtype),
               ('fs_time', time_dtype),
               ('score', np.float_)])

    # Calculate each CRPS score in turn.
    for (ix, (fs_time, time)) in enumerate(time_combs):
        # Ensure there is only a single true value for this time.
        true_mask = true_obs['time'] == time
        true_value = true_obs['value'][true_mask]
        true_count = len(true_value)
        if true_count != 1:
            msg_fmt = 'Found {} true values for {}'
            raise ValueError(msg_fmt.format(true_count, time))
        true_value = true_value[0]

        # Calculate the CRPS for this time.
        sim_mask = np.logical_and(
            sim_obs['time'] == time, sim_obs['fs_time'] == fs_time)
        samples = sim_obs['value'][sim_mask]
        score = crps_edf_scalar(true_value, samples)

        # Update the current row of the scores table.
        scores['time'][ix] = time
        scores['fs_time'][ix] = fs_time
        scores['score'][ix] = score

    return scores
