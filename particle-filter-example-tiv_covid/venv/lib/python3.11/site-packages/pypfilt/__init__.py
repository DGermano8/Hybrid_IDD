"""A bootstrap particle filter for epidemic forecasting."""

import datetime
import logging
import os
import os.path

from . import adaptive
from . import cache
from . import build
from . import pfilter
from . import io
from . import model
from . import obs
from . import sampler
from . import scenario
from . import summary
from . import time
from . import version

__package_name__ = u'pypfilt'
__author__ = u'Rob Moss'
__email__ = u'rgmoss@unimelb.edu.au'
__copyright__ = u'2014-2022, Rob Moss'
__license__ = u'BSD 3-Clause License'
__version__ = version.__version__


# Export abstract base classes from this module.
Instance = scenario.Instance
Context = build.Context
Obs = obs.Obs
Model = model.Model
OdeModel = model.OdeModel
Monitor = summary.Monitor
Table = summary.Table
Datetime = time.Datetime
Scalar = time.Scalar

load_instances = scenario.load_instances

# Prevent an error message if the application does not configure logging.
log = logging.getLogger(__name__).addHandler(logging.NullHandler())


def simulate_from_model(instance, particles=1, observations=1,
                        common_prng_seed=False):
    """
    Simulate observations from a model.

    :param instance: The scenario instance.
    :type instance: pypfilt.scenario.Instance
    :param particles: The number of particles; set this to ``None`` to use the
        number of particles defined in ``instance``.
    :param observations: The number of observations to simulate for each
        particle, at each time unit.
    :param common_prng_seed: Whether the simulated observation tables should
        use a common PRNG seed to generate the simulated observations.
    :return: A dictionary of simulated observation tables.
        The dictionary will be **empty** if no observations were simulated
        (e.g., if no time-steps were performed).
    :rtype: Dict[str, numpy.ndarray]

    .. note:: The ``instance`` **should not be reused** after calling this
        function.
        To prevent this from happening, the instance settings will be deleted.

    :Examples:

    >>> import pypfilt
    >>> import pypfilt.examples.predation
    >>> pypfilt.examples.predation.write_example_files()
    >>> config_file = 'predation.toml'
    >>> for instance in pypfilt.load_instances(config_file):
    ...     obs_tables = pypfilt.simulate_from_model(instance, particles=1)
    ...     # Print the first four simulated 'x' observations.
    ...     x_obs = obs_tables['x']
    ...     print(x_obs[:4])
    ...     # Print the first four simulated 'y' observations.
    ...     y_obs = obs_tables['y']
    ...     print(y_obs[:4])
    [(0., 1.35192613) (1., 1.54453284) (2., 1.92039984) (3., 1.21684457)]
    [(0., -0.14294339) (1.,  0.51294785) (2.,  1.14303379) (3.,  0.84072101)]
    >>> print(instance.settings)
    {}
    >>> # Remove the example files when they are no longer needed.
    >>> pypfilt.examples.predation.remove_example_files()
    """
    logger = logging.getLogger(__name__)

    if not isinstance(instance, scenario.Instance):
        msg_fmt = 'Value of type {} is not a scenario instance'
        raise ValueError(msg_fmt.format(type(instance)))

    if 'summary' not in instance.settings:
        instance.settings['summary'] = {}

    # NOTE: only overwrite/replace summary tables and monitors.
    # Leave other summary settings untouched.
    if 'tables' in instance.settings['summary']:
        del instance.settings['summary']['tables']
    if 'monitors' in instance.settings['summary']:
        del instance.settings['summary']['monitors']
    # NOTE: remove initialisation arguments for custom summary components.
    if 'init' in instance.settings['summary']:
        del instance.settings['summary']['init']

    instance.settings['summary']['tables'] = {}

    # NOTE: we need a separate table for each observation unit.
    obs_units = instance.settings.get('observations', {}).keys()
    for obs_unit in obs_units:
        instance.settings['summary']['tables'][obs_unit] = {
            'component': 'pypfilt.summary.SimulatedObs',
            'observation_unit': obs_unit,
            'common_prng_seed': common_prng_seed,
            'observations_per_particle': observations,
            'include_fs_time': False,
        }

    if particles is not None:
        instance.settings['filter']['particles'] = particles

    # To ensure that the simulation runs successfully, we have to avoid using
    # a custom summary function (such as epifx.summary.make), since they may
    # create tables that, e.g., require observations.
    instance.settings['components']['summary'] = 'pypfilt.summary.HDF5'

    # Do not load observations from disk.
    ctx = instance.build_context(obs_tables={})

    # Ensure that the output directory exists, or create it.
    io.ensure_directory_exists(ctx.settings['files']['output_directory'])

    ctx.component['summary'].initialise(ctx)

    # Empty instance.settings so that the instance cannot be reused.
    settings_keys = list(instance.settings.keys())
    for key in settings_keys:
        del instance.settings[key]

    start = ctx.settings['time']['start']
    until = ctx.settings['time']['until']
    logger.info("  {}  Estimating  from {} to {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"),
        start, until))
    result = pfilter.run(ctx, start, until, {})

    # NOTE: run() may return None if est_end < (start + dt).
    if result is None:
        return {}

    # Return the dictionary of simulated observation tables.
    return result.tables


def forecast(ctx, times, filename):
    """Generate forecasts from various times during a simulation.

    :param ctx: The simulation context.
    :type ctx: pypfilt.build.Context
    :param times: The times at which forecasts should be generated.
    :param filename: The output file to generate (can be ``None``).

    :returns: The simulation results for each forecast.
    :rtype: ~pypfilt.pfilter.Results

    :Examples:

    >>> from datetime import datetime
    >>> import pypfilt
    >>> import pypfilt.build
    >>> import pypfilt.examples.predation
    >>> pypfilt.examples.predation.write_example_files()
    >>> config_file = 'predation-datetime.toml'
    >>> fs_times = [datetime(2017, 5, 5), datetime(2017, 5, 10)]
    >>> data_file = 'output.hdf5'
    >>> for instance in pypfilt.load_instances(config_file):
    ...     context = instance.build_context()
    ...     results = pypfilt.forecast(context, fs_times, filename=data_file)
    >>> # Remove the example files when they are no longer needed.
    >>> pypfilt.examples.predation.remove_example_files()
    >>> # Remove the output file when it is no longer needed.
    >>> import os
    >>> os.remove(data_file)
    """

    if not isinstance(ctx, Context):
        msg_fmt = 'Value of type {} is not a simulation context'
        raise ValueError(msg_fmt.format(type(ctx)))

    # Ensure that there is at least one forecasting time.
    if len(times) < 1:
        raise ValueError("No forecasting times specified")

    # Ensure that forecasting times are valid time values.
    time_scale = ctx.component['time']
    times = [time_scale.parse(time) for time in times]

    start = ctx.settings['time']['start']
    end = ctx.settings['time']['until']
    if start is None or end is None:
        raise ValueError("Simulation period is not defined")

    # Ensure that the forecasting times lie within the simulation period.
    invalid_fs = [ctx.component['time'].to_unicode(t) for t in times
                  if t < start or t >= end]
    if invalid_fs:
        raise ValueError("Invalid forecasting time(s) {}".format(invalid_fs))

    logger = logging.getLogger(__name__)

    # Ensure that the output directory exists, or create it.
    io.ensure_directory_exists(ctx.settings['files']['output_directory'])

    # Initialise the summary object.
    ctx.component['summary'].initialise(ctx)

    # Generate forecasts in order from earliest to latest forecasting time.
    # Note that forecasting from the start time will duplicate the estimation
    # run (below) and is therefore redundant *if* sim['end'] is None.
    forecast_times = [t for t in sorted(times) if t >= start]

    # Identify the cache file, and remove it if instructed to do so.
    sim = cache.default(ctx, forecast_times)
    cache_file = sim['save_to']
    if ctx.settings['files']['delete_cache_file_before_forecast']:
        cache.remove_cache_file(cache_file)

    # Load the most recently cached simulation state that is consistent with
    # the current observations.
    update = cache.load_state(cache_file, ctx, forecast_times)
    if update is not None:
        for (key, value) in update.items():
            sim[key] = value

    # Update the forecasting times.
    if not sim['fs_times']:
        logger.warning("All {} forecasting times precede cached state".format(
            len(forecast_times)))
        return
    forecast_times = sim['fs_times']

    # Update the simulation period.
    if sim['start'] is not None:
        start = sim['start']
    if sim['end'] is not None:
        # Only simulate as far as the final forecasting time, then forecast.
        # Note that this behaviour may not always be desirable, so it can be
        # disabled by setting 'minimal_estimation_run' to False.
        if ctx.settings['filter']['minimal_estimation_run']:
            est_end = sim['end']
    else:
        est_end = end

    # Avoid the estimation pass when possible.
    estim_result = None
    adaptive_fits = []
    if start < est_end:
        # Check whether to use adaptive_fit() for the estimation pass.
        adaptive = ctx.get_setting(
            ['filter', 'adaptive_fit', 'enabled'],
            False)
        if adaptive:
            # NOTE: this returns a dictionary that contains the results of the
            # estimation pass ('complete') and the results of each adaptive
            # fitting pass ('adaptive_fit').
            results = adaptive_fit(ctx, filename=None,
                                   save_when=forecast_times,
                                   save_to=sim['save_to'])
            adaptive_fits = results.adaptive_fits
            estim_result = results.estimation
        else:
            # Use the standard single estimation pass.
            logger.info("  {}  Estimating  from {} to {}".format(
                datetime.datetime.now().strftime("%H:%M:%S"), start, est_end))
            estim_result = pfilter.run(ctx, start, est_end, ctx.data['obs'],
                                       history=sim['history'],
                                       save_when=forecast_times,
                                       save_to=sim['save_to'])
        # NOTE: record whether this simulation resumed from a cached state.
        if sim['start'] is not None:
            estim_result.loaded_from_cache = sim['start']
    else:
        logger.info("  {}  No estimation pass needed for {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"), est_end))

    # Ensure the times are ordered from latest to earliest.
    forecasts = {}
    for start_time in forecast_times:
        # We can reuse the history matrix for each forecast, since all of the
        # pertinent details are recorded in the summary.
        update = cache.load_state_at_time(cache_file, ctx, start_time)

        # There may not be a cached state when forecasting from the start of
        # the simulation period.
        if update is None and start_time == start:
            update = {'history': None}
        elif update is None:
            msg = 'Cache file missing entry for forecast time {}'
            raise ValueError(msg.format(start_time))

        # Check whether the particles should be resampled before forecasting.
        do_resample = ctx.get_setting(
            ['filter', 'resample', 'before_forecasting'],
            False)
        if do_resample and update['history'] is not None:
            logger.debug("  {:%H:%M:%S}  Resampling before forecast".format(
                datetime.datetime.now()))
            snapshot = update['history'].snapshot(ctx)
            pfilter.resample_ensemble(ctx, snapshot, threshold_fraction=1.0)

        # The forecast may not extend to the end of the simulation period.
        fs_end = end
        if 'max_forecast_ahead' in ctx.settings['time']:
            max_end = ctx.component['time'].add_scalar(
                start_time,
                ctx.settings['time']['max_forecast_ahead'])
            if max_end < fs_end:
                fs_end = max_end

        logger.info("  {}  Forecasting from {} to {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            ctx.component['time'].to_unicode(start_time),
            ctx.component['time'].to_unicode(fs_end)))

        fstate = pfilter.run(ctx, start_time, fs_end, {},
                             history=update['history'])
        fstate.loaded_from_cache = start_time

        forecasts[start_time] = fstate

    results = pfilter.Results(
        estimation=estim_result,
        forecasts=forecasts,
        obs=ctx.all_observations,
        adaptive_fits=adaptive_fits)

    # Save the forecasting results to disk.
    if filename is not None:
        logger.info("  {}  Saving to:  {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"), filename))
        # Save the results in the output directory.
        filepath = os.path.join(
            ctx.settings['files']['output_directory'], filename)
        ctx.component['summary'].save_forecasts(ctx, results, filepath)

    # Remove the temporary file and directory.
    sim['clean']()

    # Remove the cache file if instructed to do so.
    if ctx.settings['files']['delete_cache_file_after_forecast']:
        cache.remove_cache_file(cache_file)

    return results


def adaptive_fit(ctx, filename, save_when=None, save_to=None):
    """
    Run a series of adaptive fits over all observations.

    :param ctx: The simulation context.
    :type ctx: pypfilt.build.Context
    :param filename: The output file to generate (can be ``None``).
    :param save_when: Times at which to save the particle history matrix
        during the **final** estimation pass.
    :param save_to: The filename for saving the particle history matrix.

    :returns: The simulation results for the estimation passes.
    :rtype: ~pypfilt.pfilter.Results

    The fitting method must be defined in ``filter.adaptive_fit.method``.
    The supported methods are:

    - ``"fit_effective_fraction"``: each pass tunes the observation models to
      achieve a target effective particle fraction.
    - ``"fixed_exponents"``: each pass raises the log-likelihoods to a fixed
      exponent.
    """
    logger = logging.getLogger(__name__)

    if not isinstance(ctx, Context):
        msg_fmt = 'Value of type {} is not a simulation context'
        raise ValueError(msg_fmt.format(type(ctx)))

    method = ctx.get_setting(['filter', 'adaptive_fit', 'method'])
    if method is None:
        raise ValueError('Undefined setting: filter.adaptive_fit.method')
    valid_methods = ['fit_effective_fraction', 'fixed_exponents']
    if method not in valid_methods:
        msg = 'Invalid value for filter.adaptive_fit.method: {}'
        raise ValueError(msg.format(method))

    (start, until) = adaptive._simulation_period(ctx)

    if method == 'fit_effective_fraction':
        fits = adaptive.fit_effective_fraction(ctx, start, until)
    elif method == 'fixed_exponents':
        fits = adaptive.fixed_exponents(ctx, start, until)

    # Run an estimation pass using the final fitted ensemble.
    # By allowing the state to be saved at various times during this pass, we
    # provide an alternative to the standard estimation pass in
    # `pypfilt.forecast()`.
    final_pass = pfilter.run(ctx, start, until, ctx.data['obs'],
                             save_when=save_when, save_to=save_to)

    results = pfilter.Results(
        estimation=final_pass,
        forecasts={},
        obs=ctx.all_observations,
        adaptive_fits=fits)

    # Save the forecasting results to disk.
    if filename is not None:
        logger.info("  {}  Saving to:  {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"), filename))
        # Save the results in the output directory.
        filepath = os.path.join(
            ctx.settings['files']['output_directory'], filename)
        ctx.component['summary'].save_forecasts(ctx, results, filepath)

    return results


def fit(ctx, filename):
    """
    Run a single estimation pass over the entire simulation period.

    :param ctx: The simulation context.
    :type ctx: pypfilt.build.Context
    :param filename: The output file to generate (can be ``None``).

    :returns: The simulation results for the estimation pass.
    :rtype: ~pypfilt.pfilter.Results

    :Examples:

    >>> import pypfilt
    >>> import pypfilt.build
    >>> import pypfilt.examples.predation
    >>> pypfilt.examples.predation.write_example_files()
    >>> config_file = 'predation.toml'
    >>> data_file = 'output.hdf5'
    >>> for instance in pypfilt.load_instances(config_file):
    ...     context = instance.build_context()
    ...     results = pypfilt.fit(context, filename=data_file)
    >>> # Remove the example files when they are no longer needed.
    >>> pypfilt.examples.predation.remove_example_files()
    >>> # Remove the output file when it is no longer needed.
    >>> import os
    >>> os.remove(data_file)
    """
    if not isinstance(ctx, Context):
        msg_fmt = 'Value of type {} is not a simulation context'
        raise ValueError(msg_fmt.format(type(ctx)))

    start = ctx.settings['time']['start']
    until = ctx.settings['time']['until']
    if start is None or until is None:
        raise ValueError("Simulation period is not defined")

    logger = logging.getLogger(__name__)

    # Ensure that the output directory exists, or create it.
    io.ensure_directory_exists(ctx.settings['files']['output_directory'])

    # Initialise the summary object.
    ctx.component['summary'].initialise(ctx)

    logger.info("  {}  Estimating  from {} to {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"), start, until))
    result = pfilter.run(ctx, start, until, ctx.data['obs'])
    results = pfilter.Results(
        estimation=result,
        forecasts={},
        obs=ctx.all_observations,
        adaptive_fits=[])

    # Save the forecasting results to disk.
    if filename is not None:
        logger.info("  {}  Saving to:  {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"), filename))
        # Save the results in the output directory.
        filepath = os.path.join(
            ctx.settings['files']['output_directory'], filename)
        ctx.component['summary'].save_forecasts(ctx, results, filepath)

    return results
