"""Particle filter state cache."""

import atexit
import logging
import h5py
import numpy as np
import os.path
import signal
import tempfile
import tomli
import tomli_w

from .io import load_dataset, save_dataset
from .state import History


def default(ctx, forecast_times):
    """
    Return the default (i.e., empty) cached state.

    If no cache file is defined in
    ``ctx.settings['files']['cache_file']``, this will create a temporary
    cache file that will be automatically deleted when the process terminates
    (either normally, or in response to the SIGTERM signal).

    :param ctx: The simulation context.
    :param forecast_times: The times at which forecasts will be run.
    :returns: A dictionary with keys:

        - ``'state'``: Either a simulation state (see
          :func:`pypfilt.pfilter.run`) or ``None`` if there is no cached state
          from which to resume.
        - ``'start'``: Either the time from which to begin the simulation, or
          ``None`` if there is no cached state.
        - ``'end'``: Either the time at which to end the simulation, or
          ``None`` if there is no cached state.
        - ``'fs_times'``: The times at which forecasts should be generated.
        - ``'save_to'``: The filename for saving the particle history matrix.
        - ``'clean'``: A cleanup function to remove any temporary files, and
          which will have been registered to execute at termination.
    """
    cache_file = None
    if 'cache_file' in ctx.settings['files']:
        if ctx.settings['files']['cache_file'] is not None:
            cache_file = os.path.join(
                ctx.settings['files']['output_directory'],
                ctx.settings['files']['cache_file'])

    if cache_file is None:
        cache_file, clean_fn = temporary_cache(ctx)
    else:
        def clean_fn():
            pass

    # The default, should there be no suitable cached state.
    result = {'history': None,
              'start': None,
              'end': max(forecast_times),
              'fs_times': forecast_times,
              'save_to': cache_file,
              'clean': clean_fn}
    return result


def temporary_cache(ctx):
    logger = logging.getLogger(__name__)
    tmp_dir = tempfile.mkdtemp(dir=ctx.settings['files']['temp_directory'])
    tmp_file = os.path.join(tmp_dir, "history.hdf5")

    # Ensure these files are always deleted upon *normal* termination.
    atexit.register(__cleanup, files=[tmp_file], dirs=[tmp_dir])

    # Ensure these files are always deleted when killed by SIGTERM.
    def clean_at_terminate(signal_num, stack_frame):
        __cleanup(files=[tmp_file], dirs=[tmp_dir])
        os._exit(0)

    signal.signal(signal.SIGTERM, clean_at_terminate)

    logger.debug("Temporary file for history matrix: '{}'".format(
        tmp_file))

    def clean():
        __cleanup(files=[tmp_file], dirs=[tmp_dir])

    return (tmp_file, clean)


def __cleanup(files, dirs):
    """Delete temporary files and directories.
    This is intended for use with ``atexit.register()``.

    :param files: The list of files to delete.
    :param dirs: The list of directories to delete (*after* all of the files
        have been deleted). Note that these directories must be empty in order
        to be deleted.
    """
    logger = logging.getLogger(__name__)

    for tmp_file in files:
        if os.path.isfile(tmp_file):
            try:
                os.remove(tmp_file)
                logger.debug("Deleted '{}'".format(tmp_file))
            except OSError as e:
                msg = "Can not delete '{}': {}".format(tmp_file, e.strerror)
                logger.warning(msg)
        elif os.path.exists(tmp_file):
            logger.warning("'{}' is not a file".format(tmp_file))
        else:
            logger.debug("File '{}' already deleted".format(tmp_file))

    for tmp_dir in dirs:
        if os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
                logger.debug("Deleted '{}'".format(tmp_dir))
            except OSError as e:
                msg = "Can not delete '{}': {}".format(tmp_dir, e.strerror)
                logger.warning(msg)
        elif os.path.exists(tmp_dir):
            logger.warning("'{}' is not a directory".format(tmp_dir))
        else:
            logger.debug("Directory '{}' already deleted".format(tmp_dir))


def remove_cache_file(cache_file):
    """
    Remove the specified cache file; if the cache file does not exist, this
    has no effect and will not raise an exception.

    :param cache_file: The **full path** to the cache file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Removing cache file {} ...'.format(cache_file))

    try:
        os.remove(cache_file)
    except FileNotFoundError:
        # Don't raise an error if the file doesn't exist.
        pass


def save_state(cache_file, ctx, when, **kwargs):
    """
    Save the particle history matrix to a cache file, to allow future
    forecasting runs to resume from this point.

    :param cache_file: The name of the cache file.
    :param ctx: The simulation context.
    :param when: The current simulation time.
    :param \\**kwargs: The data sets to store in the cached state; at a
        minimum this should include ``'hist'`` (the particle history matrix)
        and ``'offset'`` (the index of the current time-step in the particle
        history matrix).

    :Examples:

    .. code:: python

       cache_file = 'cache.hdf5'
       cache.save_state(cache_file, context, current_time,
                        offset=np.int32(hist_ix),
                        hist=hist)
    """
    logger = logging.getLogger(__name__)
    with h5py.File(cache_file, 'a') as f:
        when_str = ctx.component['time'].to_unicode(when)
        logger.debug('Saving cached state for {}'.format(when_str))
        grp = f.require_group(when_str)
        # Save the history matrix state.
        history_grp = grp.require_group('history')
        ctx.component['history'].save_state(ctx, history_grp)
        # Save all of the data tables.
        data_grp = grp.require_group('data')
        __save_data_tables(ctx, f, data_grp, ctx.data)
        # Save the state of the summary object.
        sum_grp = grp.require_group('summary')
        ctx.component['summary'].save_state(ctx, sum_grp)
        # Save the state of the random number generators.
        save_rng_states(grp, 'prng_states', ctx.component['random'])


def save_rng_states(group, dataset_name, rngs):
    """
    Save random number generator states to a cached dataset.

    :param group: The HDF5 group that will contain the saved states.
    :param dataset_name: The name of the HDF5 dataset to create.
    :param rngs: A dictionary that maps names (strings) to random number
        generators.
    """
    dt = h5py.string_dtype(encoding='utf-8')
    dtype = [('name', dt), ('state', dt)]
    shape = (len(rngs),)
    dataset = group.create_dataset(dataset_name, shape, dtype=dtype)
    for (ix, (name, rng)) in enumerate(rngs.items()):
        state = tomli_w.dumps(rng.bit_generator.state)
        dataset[ix] = (name, state)


def load_rng_states(group, dataset_name, rngs):
    """
    Restore random number generator states from a cached dataset.

    :param group: The HDF5 group that contains the saved states.
    :param dataset_name: The name of the HDF5 dataset.
    :param rngs: A dictionary that maps names (strings) to random number
        generators, whose states will be updated.

    :raises ValueError: if there are missing names or additional names in the
        dataset.
    """
    dataset = group[dataset_name][()]
    names = set()
    for (name, state) in dataset:
        name = name.decode(encoding='utf-8')
        state = state.decode(encoding='utf-8')
        if name not in rngs:
            raise ValueError('Unknown PRNG {}'.format(name))
        names.add(name)
        rngs[name].bit_generator.state = tomli.loads(state)
    rng_names = set(rngs.keys())
    if rng_names != names:
        unknown = rng_names - names
        raise ValueError('No saved state for PRNGs {}'.format(unknown))


def __save_data_tables(ctx, hdf5_file, group, data, path=None):
    if path is None:
        path = []
    for (name, value) in data.items():
        if isinstance(value, dict):
            if value:
                # Nested, non-empty dictionary.
                subgroup = group.require_group(name)
                path.append(name)
                __save_data_tables(ctx, hdf5_file, subgroup, value, path)
        elif isinstance(value, np.ndarray):
            if name in group:
                del group[name]
            save_dataset(ctx.component['time'], group, name, value)
        else:
            raise ValueError('Invalid data table {}.{} has type {}'.format(
                '.'.join(path), name, type(value)))


def load_state(cache_file, ctx, forecast_times):
    """
    Load the particle history matrix from a cache file, allowing forecasting
    runs to resume at the point of the first updated/new observation.

    :param cache_file: The name of the cache file.
    :param ctx: The simulation context.
    :param forecast_times: The times at which forecasts will be run.

    :returns: Either ``None``, if there was no suitable cached state, or a
        dictionary with following keys:

        - ``'start'``: The time from which to begin the simulation.
        - ``'end'``: The time from which to end the simulation.
        - ``'state'``: A dictionary that contains the following keys:

            - ``'hist'``: The particle history matrix.
            - ``'offset'``: The index of the current time-step in the particle
              history matrix.

    Note that if the cache file already contains a suitable state for each of
    the provided forecast times, this will return a dictionary as described
    above, where the ``'start'`` and ``'end'`` values are the same (i.e.,
    there is no need to run an estimation pass).
    """
    logger = logging.getLogger(__name__)
    logger.debug('Searching for cached state in {}'.format(cache_file))

    if not os.path.exists(cache_file):
        logger.debug("Missing cache file: '{}'".format(cache_file))
        return None

    try:
        with h5py.File(cache_file, 'r') as f:
            logger.debug("Reading cache file: '{}'".format(cache_file))
            return __find_most_recent_time(ctx, forecast_times, f)
    except IOError:
        logger.debug("Could not read cache file: '{}'".format(cache_file))
        return None


def load_state_at_time(cache_file, ctx, when):
    """
    Load the particle history matrix from a cache file at a specific
    simulation time.

    :param cache_file: The name of the cache file.
    :param ctx: The simulation context.
    :param when: The simulation time at which the particle history matrix was
        cached.

    :returns: Either ``None``, if there was no cached result for the given
        simulation time, or a dictionary as per :func:`load_state`.
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(cache_file):
        logger.debug("Missing cache file: '{}'".format(cache_file))
        return None

    time = ctx.component['time']
    summary = ctx.component['summary']
    model = ctx.component['model']

    try:
        with h5py.File(cache_file, 'r') as f:
            logger.debug("Reading cache file: '{}'".format(cache_file))
            when_str = time.to_unicode(when)
            if when_str not in f:
                logger.debug('Cache file has no entry for "{}"'
                             .format(when_str))
            group = f[when_str]
            history = History.load_state(ctx, group['history'])
            result = {
                'start': when,
                'history': history,
            }
            summary.load_state(ctx, group['summary'])
            # Restore the state of the random number generators.
            load_rng_states(group, 'prng_states', ctx.component['random'])
            # Inform the model that we're resuming from a cached state.
            model.resume_from_cache(ctx)
            return result
    except IOError:
        logger.debug("Could not read cache file: '{}'".format(cache_file))
        return None


def __find_most_recent_time(ctx, forecast_times, hdf5_file):
    time = ctx.component['time']
    summary = ctx.component['summary']
    model = ctx.component['model']

    logger = logging.getLogger(__name__)

    # Starting from the earliest forecasting time, identify the forecasting
    # times for which there is a matching cached state. This search stops when
    # it encounters the first forecasting time for which there is no matching
    # cached state, because the estimation pass will need to start from the
    # previous forecasting time.
    forecast_times = sorted(forecast_times)
    forecast_matches = []
    for fs_time in forecast_times:
        group_name = time.to_unicode(fs_time)
        if group_name not in hdf5_file:
            logger.debug('No cached state for {}'.format(group_name))
            break
        group = hdf5_file[group_name]
        if __data_match(time, ctx.data, group['data'], fs_time):
            logger.debug('Matching cached state for {}'.format(group_name))
            forecast_matches.append(fs_time)
        else:
            logger.debug('No matching cached state for {}'.format(group_name))
            break

    # If we have a matching cached state for the earliest forecasting time,
    # and possibly for subsequent forecasting times, we can start the
    # estimation pass at the latest of these forecasting times.
    #
    # If there is a matching cached state for all of the forecasting times,
    # the 'start' and 'end' times will be identical and the estimation pass
    # will be avoided entirely.
    if forecast_matches:
        cache_time = forecast_matches[-1]
        group_name = time.to_unicode(cache_time)
        logger.debug('Using cached state at {}'.format(group_name))
        group = hdf5_file[group_name]
        history = History.load_state(ctx, group['history'])
        result = {
            'start': cache_time,
            'end': max(forecast_times),
            'history': history,
        }
        summary.load_state(ctx, group['summary'])
        # Restore the state of the random number generators.
        load_rng_states(group, 'prng_states', ctx.component['random'])
        # Inform the model that we're resuming from a cached state.
        model.resume_from_cache(ctx)
        return result

    # Build a table that maps the cache time to the group name.
    # NOTE: the use of 'obs' to identify observation tables.
    cache_table = {time.from_unicode(group): group for group in hdf5_file
                   if group != 'obs'}
    # Sort cached times from newest to oldest.
    cache_times = reversed(sorted([time for time in cache_table.keys()
                                   if time <= min(forecast_times)]))

    # Starting with the newest time, find the first cached state that is
    # consistent with all of the data tables in ctx.data.
    for cache_time in cache_times:
        group_name = cache_table[cache_time]
        group = hdf5_file[group_name]
        if __data_match(time, ctx.data, group['data'], cache_time):
            logger.debug('Using cached state for {}'.format(group_name))
            history = History.load_state(ctx, group['history'])
            result = {
                'start': cache_time,
                'end': max(forecast_times),
                'history': history,
            }
            summary.load_state(ctx, group['summary'])
            # Restore the state of the random number generators.
            load_rng_states(group, 'prng_states', ctx.component['random'])
            # Inform the model that we're resuming from a cached state.
            model.resume_from_cache(ctx)
            return result

    logger.debug('Found no cached state')
    return None


def __data_match(time, ctx_data, cache_data, cache_time):
    # NOTE: because __save_data_tables() only creates sub-groups when it
    # encounters a *non-empty* dictionary, we need to ignore empty
    # dictionaries in the context data when comparing keys here.
    # A straight `ctx_data[k] != {}` comparison raises warnings when the value
    # is a numpy array, so we first need to check whether the value is a
    # dictionary.
    ctx_keys = set(k for k in ctx_data.keys()
                   if (not isinstance(ctx_data[k], dict))
                   or ctx_data[k] != {})
    data_keys = set(cache_data.keys())
    if ctx_keys != data_keys:
        return False

    for key in ctx_keys:
        ctx_val = ctx_data[key]
        cache_val = cache_data[key]
        if isinstance(ctx_val, dict) and isinstance(cache_val, h5py.Group):
            # Compare nested group of data tables.
            if not __data_match(time, ctx_val, cache_val, cache_time):
                return False
        elif (isinstance(ctx_val, np.ndarray)
              and isinstance(cache_val, h5py.Dataset)):
            # Compare data tables using native time values.
            cache_tbl = load_dataset(time, cache_val)

            # Detect whether we are comparing observation tables.
            # NOTE: the use of 'obs' to identify observation tables.
            obs_prefix = '/{}/data/obs/'.format(cache_time)
            is_obs = cache_val.name.startswith(obs_prefix)

            if is_obs:
                # NOTE: we only want to compare observations up to the cache
                # time.
                ctx_sub = ctx_val[ctx_val['time'] <= cache_time]
                cache_sub = cache_tbl[cache_tbl['time'] <= cache_time]
                is_match = np.array_equal(ctx_sub, cache_sub)
            else:
                is_match = np.array_equal(ctx_val, cache_tbl)

            if not is_match:
                return False
        else:
            # NOTE: mismatch between context and cache group structure.
            return False

    return True
