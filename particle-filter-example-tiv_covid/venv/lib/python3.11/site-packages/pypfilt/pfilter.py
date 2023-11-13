"""Particle filter core: simulate time-steps and adjust particle weights."""

from dataclasses import dataclass, field
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from . import cache, resample
from . import obs as obs_mod
from . import state as state_mod
from .summary import Table


@dataclass
class Result:
    """
    The results obtained from running a particle filter pass.

    :param settings: The scenario instance settings.
    :type settings: Dict[str, Any]
    :param history: The simulation history component.
    :type history: ~pypfilt.state.History
    :param tables: The summary statistic tables.
    :type tables: Dict[str, numpy.ndarray]
    :param loaded_from_cache: The time (if any) from which the pass began,
        starting from a cached simulation state.
    :type loaded_from_cache: Optional[Any]
    :param metadata: A dictionary that can be used to record additional
        information about these results.
    :type metadata: Dict[Any, Any]
    """
    settings: Dict[str, Any]
    history: state_mod.History
    tables: Dict[str, Table]
    loaded_from_cache: Optional[Any] = None
    metadata: Dict[Any, Any] = field(default_factory=dict)


@dataclass
class Results:
    """
    The results obtained from running one or more particle filter passes.

    :param estimation: The results of the estimation pass (if performed).
    :type estimation: Optional[Result]
    :param forecasts: The results of each forecasting pass, indexed by
        forecasting time.
    :type forecasts: Dict[Any, Result]
    :param adaptive_fits: The results of the each adaptive fitting pass,
        indexed by exponent.
    :type adaptive_fits: Dict[float, Result]
    :param obs: The available observations.
    :type obs: List[Dict[str, Any]]
    :param metadata: A dictionary that can be used to record additional
        information about these results.
    :type metadata: Dict[Any, Any]
    """
    estimation: Optional[Result] = None
    forecasts: Dict[Any, Result] = field(default_factory=dict)
    adaptive_fits: Dict[float, Result] = field(default_factory=dict)
    obs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[Any, Any] = field(default_factory=dict)

    def forecast_times(self):
        """Return the times for which forecasts were generated."""
        return self.forecasts.keys()

    def adaptive_fit_exponents(self):
        """Return the exponents for which adaptive fits were generated."""
        return self.adaptive_fits.keys()


def reweight_partition(ctx, snapshot, logs, partition):
    """Adjust particle weights in response to some observation(s).

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param logs: The log-likelihoods of the available observation(s) for each
        particle.
    :type logs: numpy.ndarray
    :param partition: The ensemble partition to reweight.
    :type partition: Dict[str, Any]

    :returns: The **effective** number of particles (i.e., accounting for
        weights).
    :rtype: float
    """
    logger = logging.getLogger(__name__)

    p_ixs = partition['slice']
    net_w = partition['weight']
    net_w_sq = np.square(net_w)

    # Calculate the effective number of particles, prior to reweighting.
    prev_eff = net_w_sq / sum(w * w for w in snapshot.weights[p_ixs])
    # Update the current weights.
    new_weights = snapshot.weights[p_ixs] * np.exp(logs[p_ixs])
    ws_sum = np.sum(sorted(new_weights))
    if ws_sum == 0:
        fail = ctx.settings['filter']['reweight_or_fail']
        msg = 'Updated particle weights sum to zero at {} for {}'.format(
            snapshot.time, ctx.scenario_id)
        if fail:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            logger.warning('Retaining previous particle weights')
            ws_sum = np.sum(sorted(snapshot.weights[p_ixs]))
    else:
        # Update and renormalise the weights.
        snapshot.weights[p_ixs] = new_weights
        snapshot.weights[p_ixs] *= net_w / ws_sum

    if np.any(np.isnan(snapshot.weights[p_ixs])):
        # Either the new weights were all zero, or every new non-zero weight
        # is associated with a particle whose previous weight was zero.
        nans = np.sum(np.isnan(snapshot.weights[p_ixs]))
        raise ValueError("{} NaN weights; ws_sum = {}".format(nans, ws_sum))
    # Determine whether resampling is required.
    num_eff = net_w_sq / sum(snapshot.weights[p_ixs] ** 2)

    # Detect when the effective number of particles has greatly decreased.
    eff_decr = num_eff / prev_eff
    if (eff_decr < 0.1):
        # Note: this could be mitigated by replacing the weights with their
        # square roots (for example) until the decrease is sufficiently small.
        logger.debug("Effective particles decreased by {}".format(eff_decr))

    return num_eff


def reweight_ensemble(ctx, snapshot, logs):
    """Reweight the particles in each non-reservoir partition of the ensemble.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param logs: The log-likelihoods of the available observation(s) for each
        particle.
    :type logs: numpy.ndarray
    """
    # Scale the log-likelihoods so that the maximum is 0 (i.e., has a
    # likelihood of 1) to increase the chance of smaller likelihoods being
    # within the range of double-precision floating-point.
    logs = logs - np.max(logs)

    # Check whether particle weights should be updated.
    do_reweight = ctx.get_setting(
        ['filter', 'reweight', 'enabled'],
        True)

    if not do_reweight:
        # Update the net log-likelihood of each particle.
        key = 'net_log_llhd'
        if key in ctx.data:
            ctx.data[key] += logs
        else:
            ctx.data[key] = logs
        ctx.data[key] -= np.max(ctx.data[key])
        return

    # Update particle weights in each partition.
    partitions = ctx.settings['filter']['partition']
    for partition in partitions:
        # Do not update reservoir particle weights.
        if partition['reservoir']:
            continue

        reweight_partition(ctx, snapshot, logs, partition)


def __log_step(ctx, when, do_resample, num_eff=None):
    """Log the state of the particle filter when an observation is made or
    when particles have been resampled.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param when: The current simulation time.
    :param do_resample: Whether particles were resampled at this time-step.
    :type do_resample: bool
    :param num_eff: The effective number of particles (default is ``None``).
    :type num_eff: float
    """
    logger = logging.getLogger(__name__)
    resp = {True: 'Y', False: 'N'}
    if num_eff is not None:
        logger.debug('{} RS: {}, #px: {:7.1f}'.format(
            ctx.component['time'].to_unicode(when), resp[do_resample],
            num_eff))
    elif do_resample:
        logger.debug('{} RS: {}'.format(
            ctx.component['time'].to_unicode(when), resp[do_resample]))


def resample_ensemble(ctx, snapshot, threshold_fraction=None):
    """Resample the particles in each non-reservoir partition of the ensemble.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param threshold_fraction: The resampling threshold (default: the value of
        the ``filter.resample.threshold`` setting).
    """
    when = snapshot.time
    curr = snapshot.vec
    resample_mask = np.zeros(curr.size, dtype=np.bool_)

    # Check whether resampling should be performed.
    do_resample = ctx.get_setting(
        ['filter', 'resample', 'enabled'],
        True)
    if not do_resample:
        return resample_mask

    if threshold_fraction is None:
        threshold_fraction = ctx.settings['filter']['resample']['threshold']
    partitions = ctx.settings['filter']['partition']

    for partition in partitions:
        # Do not resample reservoir particle weights.
        if partition['reservoir']:
            continue

        mask = partition['mask']
        ixs = partition['slice']

        # Calculate the effective number of particles.
        net_w_sq = np.square(partition['weight'])
        num_eff = net_w_sq / sum(snapshot.weights[ixs] ** 2)
        eff_frac = num_eff / partition['particles']
        if eff_frac >= threshold_fraction:
            __log_step(ctx, when, False, num_eff)
            continue

        ctx.call_event_handlers('before_resample', ctx, when, curr[ixs])
        # Check whether there are reservoir particles to provide.
        res_frac = 0.0
        res_px = None
        res_ix = partition.get('reservoir_ix')
        ix_offset = ixs.start
        res_offset = 0
        if res_ix is not None:
            res_frac = partition['reservoir_fraction']
            res_ixs = partitions[res_ix]['slice']
            res_px = curr[res_ixs]
            res_offset = res_ixs.start

        # NOTE: indexing with `mask` returns a *copy*, not a *view*.
        resample.resample(ctx, curr[ixs], net_weight=partition['weight'],
                          ix_offset=ix_offset,
                          res_px=res_px, res_frac=res_frac,
                          res_offset=res_offset)
        ctx.call_event_handlers('after_resample', ctx, when, curr[ixs])
        resample_mask = resample_mask | mask
        __log_step(ctx, when, True, partition['particles'])

    return resample_mask


def step(ctx, snapshot, time_step, step_obs, is_fs):
    """Perform a single time-step for every particle.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param time_step: The time-step details.
    :type time_step: ~pypfilt.time.TimeStep
    :param step_obs: The list of observations for this time-step.
    :param is_fs: Indicate whether this is a forecasting simulation (i.e., no
        observations).
        For deterministic models it is useful to add some random noise when
        estimating, to allow identical particles to differ in their behaviour,
        but this is not desirable when forecasting.

    :return: a Boolean array that identifies particles which were resampled.
    :rtype: numpy.ndarray
    """
    # Define the particle ordering, which may be updated by ``resample``.
    # This must be defined before we can use `snapshot.back_n_steps()`.
    curr = snapshot.vec
    assert snapshot.time == time_step.end
    curr['prev_ix'] = np.arange(ctx.settings['filter']['particles'])
    prev = snapshot.back_n_steps(1)

    # Step each particle forward by one time-step.
    curr_sv = curr['state_vec']
    prev_sv = prev['state_vec']
    ctx.component['model'].update(ctx, time_step, is_fs, prev_sv, curr_sv)

    # Copy the particle weights from the previous time-step.
    # These will be updated by ``reweight`` as necessary.
    curr['weight'] = prev['weight']

    # Update sample lookup columns, if present.
    if curr.dtype.names is not None and 'lookup' in curr.dtype.names:
        curr['lookup'] = prev['lookup']

    # Account for observations, if any.
    if step_obs:
        # Calculate the log-likelihood of obtaining the given observation, for
        # each particle.
        logs = obs_mod.log_llhd(ctx, snapshot, step_obs)

        # Scale the likelihoods by the specified exponent.
        exponent = ctx.get_setting(['filter', 'reweight', 'exponent'], 1.0)
        if exponent < 0 or exponent > 1:
            msg = 'Invalid reweighting exponent: {}'
            raise ValueError(msg.format(exponent))
        logs *= exponent

        # Update particle weights in each partition.
        reweight_ensemble(ctx, snapshot, logs)

        # Resample particles in each partition, as required.
        resample_mask = resample_ensemble(ctx, snapshot)
    else:
        # Indicate that no particles were resampled.
        resample_mask = np.zeros(curr.size, dtype=np.bool_)

    return resample_mask


def run(ctx, start, end, obs_tables, history=None,
        save_when=None, save_to=None):
    """Run the particle filter against any number of data streams.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param start: The start of the simulation period.
    :param end: The (**exclusive**) end of the simulation period.
    :param obs_tables: A dictionary of observation tables.
    :param history: The (optional) history matrix state from which to resume.
    :param save_when: Times at which to save the particle history matrix.
    :param save_to: The filename for saving the particle history matrix.

    :returns: The resulting simulation state, which contains the simulation
        settings, the :class:`~pypfilt.state.History` component, and the
        summary statistic tables.
        If no time-steps were performed, this returns ``None``.
    :rtype: Optional[Result]
    """
    # Record the start and end of this simulation.
    ctx.settings['time']['sim_start'] = start
    ctx.settings['time']['sim_until'] = end

    sim_time = ctx.component['time']
    sim_time.set_period(
        start, end, ctx.settings['time']['steps_per_unit'])
    steps = sim_time.with_observation_tables(ctx, obs_tables)

    # Determine whether this is a forecasting run, by checking whether there
    # are any observation tables.
    is_fs = not obs_tables

    # Create the history component and store it in the simulation context.
    # We allow the history matrix to be provided in order to allow, e.g., for
    # forecasting from any point in a completed simulation.
    if history is None:
        history = state_mod.History(ctx)
    ctx.component['history'] = history

    # Allocate space for the summary statistics.
    summary = ctx.component['summary']
    summary.allocate(ctx, forecasting=is_fs)

    # Define key time-step loop variables.
    # The start of the next interval that should be summarised.
    win_start = start
    # The beginning of the current time-step.
    prev_time = start

    # Simulate each time-step.
    # NOTE: the first time-step is number 1 and updates history.matrix[1] from
    # history.matrix[0].
    for (step_num, time_step, obs) in steps:
        history.set_time_step(step_num, time_step.end)
        # Check whether the end of the history matrix has been reached.
        # If so, shift the sliding window forward in time.
        if history.reached_window_end():
            # Calculate summary statistics in blocks.
            # If most_recent is None, no time-steps have been simulated.
            # This can occur when (for example) a forecasting simulation has
            # started at the final time-step in the history matrix.
            # The correct response is to only calculate summary statistics for
            # this single time-step.
            window = history.summary_window(ctx, win_start, prev_time)
            summary.summarise(ctx, window)

            # NOTE: win_start is the start of the next interval that will be
            # summarised. Since the current time-step is not evaluated until
            # after the above call to summary.summarise(), the next summary
            # window should start at this time-step.
            win_start = time_step.end

            # Shift the moving window so that we can continue the simulation.
            shift = (ctx.settings['filter']['history_window']
                     * ctx.settings['time']['steps_per_unit'])
            history.shift_window_back(shift)

        # Simulate the current time-step.
        snapshot = history.snapshot(ctx, time_step.end)
        resampled = step(ctx, snapshot, time_step, obs, is_fs)
        # Record whether the particles were resampled at this time-step.
        history.set_resampled(resampled)

        # Check whether to save the particle history matrix to disk.
        # NOTE: the summary object may not have summarised the model state
        # recently, or even at all if we haven't reached the end of the
        # sliding window at least once.
        if save_when is not None and save_to is not None:
            if time_step.end in save_when:
                # First, summarise up to the previous time-step.
                # NOTE: we do not want to summarise the current time-step,
                # because simulations that resume from this saved state will
                # begin summarising from their initial state, which is this
                # current time-step. So if the summary window starts at the
                # current time-step, we should not summarise before saving.
                if win_start < time_step.end:
                    window = history.summary_window(ctx, win_start, prev_time)
                    summary.summarise(ctx, window)

                # Update the start of the next summary interval.
                win_start = time_step.end

                # Note: we only need to save the current matrix block!
                cache.save_state(save_to, ctx, time_step.end)

        # Finally, update loop variables.
        prev_time = time_step.end

    if history.index is None:
        # There were no time-steps, so return nothing.
        return None

    # Calculate summary statistics for the remaining time-steps.
    # There will always be at least one time-step to summarise.
    window = history.summary_window(ctx, win_start, prev_time)
    summary.summarise(ctx, window)

    # Return the complete simulation state.
    return Result(settings=ctx.settings, history=history,
                  tables=summary.get_stats())
