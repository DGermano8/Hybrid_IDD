"""Provide a range of adaptive fitting methods."""

import datetime
import logging
import numpy as np

from . import io
from . import pfilter
from .build import Context


def _simulation_period(ctx):
    """
    Return the start and end of the simulation period for estimation passes.
    """
    if not isinstance(ctx, Context):
        msg_fmt = 'Value of type {} is not a simulation context'
        raise ValueError(msg_fmt.format(type(ctx)))

    start = ctx.settings['time']['start']
    until = ctx.settings['time']['until']
    if start is None or until is None:
        raise ValueError("Simulation period is not defined")
    if ctx.settings['filter']['minimal_estimation_run']:
        final_obs = [max(table['time'])
                     for table in ctx.data['obs'].values()]
        until = max(final_obs)

    return (start, until)


def fit_effective_fraction(ctx, start, until):
    """
    Return a series of fits where each pass tunes the observation models to
    achieve a target effective particle fraction.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param start: The start of the simulation period.
    :param until: The (**exclusive**) end of the simulation period.
    :rtype: Dict[float, ~pypfilt.pfilter.Result]
    """
    logger = logging.getLogger(__name__)

    target_eff_frac = ctx.get_setting(
        ['filter', 'adaptive_fit', 'target_effective_fraction'])
    bisect_tolerance = ctx.get_setting(
        ['filter', 'adaptive_fit', 'exponent_tolerance'])
    num_px = ctx.settings['filter']['particles']

    if target_eff_frac is None:
        msg = 'Undefined setting: filter.adaptive_fit.{}'
        raise ValueError(msg.format('target_effective_fraction'))
    if bisect_tolerance is None:
        msg = 'Undefined setting: filter.adaptive_fit.{}'
        raise ValueError(msg.format('exponent_tolerance'))

    def frac_eff(log_llhds):
        net_w = sum(np.exp(log_llhds))
        net_w_sq = np.square(net_w)
        num_eff = net_w_sq / sum(np.exp(log_llhds) ** 2)
        return num_eff / num_px

    def bisect_exponent(lower, upper):
        logs = ctx.data['net_log_llhd']
        frac_lower = frac_eff(lower * logs)
        frac_upper = frac_eff(upper * logs)

        if frac_lower <= target_eff_frac:
            msg = (
                '  {}  Minimum exponent {} does not exceed target threshold;'
                ' effective fraction is {}')
            logger.info(msg.format(
                datetime.datetime.now().strftime("%H:%M:%S"),
                lower, frac_lower))
            return lower + bisect_tolerance
        elif frac_upper >= target_eff_frac:
            msg = (
                '  {}  Maximum exponent {} meets target threshold;'
                ' effective fraction is {}')
            logger.info(msg.format(
                datetime.datetime.now().strftime("%H:%M:%S"),
                upper, frac_lower))
            return upper

        def middle(l, u):
            if u < l + bisect_tolerance:
                msg = 'Bisection tolerance exceeded at {}, {}'
                logger.debug(msg.format(l, u))
                return None
            else:
                return 0.5 * (l + u)

        mid = middle(lower, upper)
        while mid is not None:
            frac_mid = frac_eff(mid * logs)
            if frac_mid <= target_eff_frac:
                upper = mid
            if frac_mid >= target_eff_frac:
                lower = mid
            mid = middle(lower, upper)

        return upper

    fixed_particles = {
        'filter': {
            'reweight': {'enabled': False},
            'resample': {'enabled': False},
        },
    }

    logger.info("  {}  Estimating  from {} to {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"), start, until))

    # Initialise the summary object.
    ctx.component['summary'].initialise(ctx)
    # Ensure that the output directory exists, or create it.
    io.ensure_directory_exists(ctx.settings['files']['output_directory'])

    fits = {}
    previous_exponent = 0
    while previous_exponent < 1:
        with ctx.override_settings(fixed_particles):
            result = pfilter.run(ctx, start, until, ctx.data['obs'])

        # Find the exponent in (previous_exponent, 1.0] that
        # (approximately) satisfies the target effective fraction.
        new_exponent = bisect_exponent(previous_exponent, 1.0)
        logger.info('  {}  Adaptive exponent is {}'.format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            new_exponent))
        logger.info('  {}  Effective fraction is {}'.format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            frac_eff(new_exponent * ctx.data['net_log_llhd'])))

        history = ctx.component['history']
        snapshot = history.snapshot(ctx, until)

        # Reweight the particles.
        logs = new_exponent * ctx.data['net_log_llhd']
        pfilter.reweight_ensemble(ctx, snapshot, logs)

        # Resample the particles and record this in the resampled column.
        resample_mask = pfilter.resample_ensemble(
            ctx, snapshot, threshold_fraction=1.0)
        history.set_resampled(resample_mask)

        # NOTE: update the model prior samples.
        final_state = snapshot.state_vec
        for name in final_state.dtype.names:
            ctx.data['prior'][name] = final_state[name]

        if new_exponent in fits:
            raise ValueError(f'Duplicate exponent {new_exponent}')
        fits[new_exponent] = result
        del ctx.data['net_log_llhd']
        previous_exponent = new_exponent

    return fits


def fixed_exponents(ctx, start, until):
    """
    Return a series of fits where the likelihoods are raised to a fixed
    sequence of exponents.

    :param ctx: The simulation parameters.
    :type ctx: ~pypfilt.build.Context
    :param start: The start of the simulation period.
    :param until: The (**exclusive**) end of the simulation period.
    :rtype: Dict[float, ~pypfilt.pfilter.Result]
    """
    logger = logging.getLogger(__name__)

    delta = ctx.get_setting(['filter', 'adaptive_fit', 'exponent_step'])
    if delta is None:
        msg = 'Undefined setting: filter.adaptive_fit.{}'
        raise ValueError(msg.format('exponent_step'))

    try:
        if len(delta) > 0:
            exponents = list(delta)
            if exponents[-1] != 1:
                exponents.append(1)
                exponents = np.array(exponents)
        else:
            raise ValueError('Empty exponent_step sequence')
    except TypeError:
        exponents = np.linspace(delta, 1.0, num=int(np.rint(1 / delta)))

    if any(exponents <= 0):
        raise ValueError('Cannot have exponents less than or equal to zero')
    if any(exponents > 1):
        raise ValueError('Cannot have exponents greater than one')
    if any(np.diff(exponents <= 0)):
        raise ValueError('Exponents must be strictly increasing')

    logger.info("  {}  Estimating  from {} to {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"), start, until))

    # Initialise the summary object.
    ctx.component['summary'].initialise(ctx)
    # Ensure that the output directory exists, or create it.
    io.ensure_directory_exists(ctx.settings['files']['output_directory'])

    fits = {}
    for current_exponent in exponents:
        # Store the exponent so that reweight() can retrieve it.
        ctx.settings['filter']['reweight']['exponent'] = current_exponent

        logger.info("  {}  Estimating  with exponent {}".format(
            datetime.datetime.now().strftime("%H:%M:%S"), current_exponent))
        result = pfilter.run(ctx, start, until, ctx.data['obs'])

        # Resample the particles and record this in the resampled column.
        history = ctx.component['history']
        snapshot = history.snapshot(ctx, until)
        resample_mask = pfilter.resample_ensemble(
            ctx, snapshot, threshold_fraction=1.0)
        history.set_resampled(resample_mask)

        # NOTE: update the model prior samples.
        final_state = snapshot.state_vec
        for name in final_state.dtype.names:
            ctx.data['prior'][name] = final_state[name]

        if current_exponent in fits:
            raise ValueError(f'Duplicate exponent {current_exponent}')
        fits[current_exponent] = result

    return fits
