"""
Examples of dynamical system models formulated by Edward Lorenz.
"""

import inspect
import numpy as np
import scipy.stats

from ..model import OdeModel
from ..obs import Univariate


class Lorenz63(OdeModel):
    r"""
    The Lorenz-63 system:

    .. math::

       \frac{dx}{dt} &= \sigma (y - x) \\
       \frac{dy}{dt} &= x (\rho - z) - y \\
       \frac{dz}{dt} &= xy - \beta z \\
       x_t &= [\sigma, \rho, \beta, x, y, z]^T

    This system exhibits chaotic behaviour in the neighbourhood of
    :math:`\sigma = 10`, :math:`\rho = 28`, :math:`\beta = \frac{8}{3}`.
    """

    def field_types(self, ctx):
        r"""
        Define the state vector :math:`[\sigma, \rho, \beta, x, y, z]^T`.
        """
        return [
            ('sigma', float), ('rho', float), ('beta', float),
            ('x', float), ('y', float), ('z', float),
        ]

    def d_dt(self, time, xt, ctx, is_forecast):
        """
        The right-hand side of the ODE system.

        :param time: The current time.
        :param xt: The particle state vectors.
        :param ctx: The simulation context.
        :param is_forecast: True if this is a forecasting simulation.
        """
        rates = np.zeros(xt.shape, xt.dtype)
        rates['x'] = xt['sigma'] * (xt['y'] - xt['x'])
        rates['y'] = xt['x'] * (xt['rho'] - xt['z']) - xt['y']
        rates['z'] = xt['x'] * xt['y'] - xt['beta'] * xt['z']
        return rates

    def can_smooth(self):
        """Indicate which state vector fields can be smoothed."""
        return {'sigma', 'rho', 'beta', 'x', 'y', 'z'}


class ObsLorenz63(Univariate):
    r"""
    An observation model for the Lorenz-63 system:

    .. math::

        y_t \sim N(\mu = x_t, \sigma = 1.5)

    The observation unit **must** be the name of a field in the state vector.
    For example, the observation unit must be ``"y"`` to observe :math:`y(t)`:

    .. code-block:: toml

       [observations.y]
       model = "pypfilt.examples.lorenz.ObsLorenz63"
    """
    def distribution(self, ctx, snapshot):
        expect = snapshot.state_vec[self.unit]
        return scipy.stats.norm(loc=expect, scale=1.5)


def lorenz63_simulate_toml():
    """
    A scenario for the :class:`Lorenz63` model, which can be used for
    simulating observations.

    :return: The scenario definition, represented as a TOML string.
    :rtype: str
    """
    return inspect.cleandoc("""
    [components]
    model = "pypfilt.examples.lorenz.Lorenz63"
    time = "pypfilt.Scalar"
    sampler = "pypfilt.sampler.LatinHypercube"
    summary = "pypfilt.summary.HDF5"

    [time]
    start = 0.0
    until = 25.0
    steps_per_unit = 10
    summaries_per_unit = 10

    [prior]
    sigma = { name = "constant", args.value = 10 }
    rho = { name = "constant", args.value = 28 }
    beta = { name = "constant", args.value = 2.66667 }
    x = { name = "constant", args.value = 1 }
    y = { name = "constant", args.value = 1 }
    z = { name = "constant", args.value = 1 }

    [observations.x]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [observations.y]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [observations.z]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [filter]
    particles = 500
    prng_seed = 2001
    history_window = -1
    resample.threshold = 0.25

    [scenario.simulate]
    """) + '\n'


def lorenz63_forecast_toml():
    """
    A scenario for the :class:`Lorenz63` model, which can be used for
    forecasting.

    :return: The scenario definition, represented as a TOML string.
    :rtype: str
    """
    return inspect.cleandoc("""
    [components]
    model = "pypfilt.examples.lorenz.Lorenz63"
    time = "pypfilt.Scalar"
    sampler = "pypfilt.sampler.LatinHypercube"
    summary = "pypfilt.summary.HDF5"

    [time]
    start = 0.0
    until = 25.0
    steps_per_unit = 10
    summaries_per_unit = 10

    [prior]
    sigma = { name = "constant", args.value = 10 }
    rho = { name = "constant", args.value = 28 }
    beta = { name = "constant", args.value = 2.66667 }
    x = { name = "uniform", args.loc = -5, args.scale = 10 }
    y = { name = "uniform", args.loc = -5, args.scale = 10 }
    z = { name = "uniform", args.loc = -5, args.scale = 10 }

    [observations.x]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-x.ssv"

    [observations.y]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-y.ssv"

    [observations.z]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-z.ssv"

    [summary.tables]
    forecasts.component = "pypfilt.summary.PredictiveCIs"
    forecasts.credible_intervals = [50, 60, 70, 80, 90, 95]

    [filter]
    particles = 500
    prng_seed = 2001
    history_window = -1
    resample.threshold = 0.25

    [scenario.forecast]
    """) + '\n'


def lorenz63_forecast_regularised_toml():
    """
    A scenario for the :class:`Lorenz63` model, which can be used for
    forecasting, and enabled post-regularisation.

    :return: The scenario definition, represented as a TOML string.
    :rtype: str
    """
    return inspect.cleandoc("""
    [components]
    model = "pypfilt.examples.lorenz.Lorenz63"
    time = "pypfilt.Scalar"
    sampler = "pypfilt.sampler.LatinHypercube"
    summary = "pypfilt.summary.HDF5"

    [time]
    start = 0.0
    until = 25.0
    steps_per_unit = 10
    summaries_per_unit = 10

    [prior]
    sigma = { name = "constant", args.value = 10 }
    rho = { name = "constant", args.value = 28 }
    beta = { name = "constant", args.value = 2.66667 }
    x = { name = "uniform", args.loc = -5, args.scale = 10 }
    y = { name = "uniform", args.loc = -5, args.scale = 10 }
    z = { name = "uniform", args.loc = -5, args.scale = 10 }

    [summary.tables]
    forecasts.component = "pypfilt.summary.PredictiveCIs"
    forecasts.credible_intervals = [50, 60, 70, 80, 90, 95]

    [observations.x]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-x.ssv"

    [observations.y]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-y.ssv"

    [observations.z]
    model = "pypfilt.examples.lorenz.ObsLorenz63"
    file = "lorenz63-z.ssv"

    [filter]
    particles = 500
    prng_seed = 2001
    history_window = -1
    resample.threshold = 0.25
    regularisation.enabled = true

    [filter.regularisation.bounds]
    x = { min = -50, max = 50 }
    y = { min = -50, max = 50 }
    z = {}

    [scenario.forecast_regularised]
    """) + '\n'


def lorenz63_all_scenarios_toml():
    """
    All example scenarios for the :class:`Lorenz63` model.

    :return: The scenario definitions, represented as a TOML string.
    :rtype: str
    """
    return inspect.cleandoc("""
    [components]
    model = "pypfilt.examples.lorenz.Lorenz63"
    time = "pypfilt.Scalar"
    sampler = "pypfilt.sampler.LatinHypercube"
    summary = "pypfilt.summary.HDF5"

    [time]
    start = 0.0
    until = 25.0
    steps_per_unit = 10
    summaries_per_unit = 10

    [prior]
    sigma = { name = "constant", args.value = 10 }
    rho = { name = "constant", args.value = 28 }
    beta = { name = "constant", args.value = 2.66667 }

    [observations.x]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [observations.y]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [observations.z]
    model = "pypfilt.examples.lorenz.ObsLorenz63"

    [filter]
    particles = 500
    prng_seed = 2001
    history_window = -1
    resample.threshold = 0.25
    regularisation.enabled = true

    [scenario.simulate]
    prior.x = { name = "constant", args.value = 1 }
    prior.y = { name = "constant", args.value = 1 }
    prior.z = { name = "constant", args.value = 1 }

    [scenario.forecast]
    prior.x = { name = "uniform", args.loc = -5, args.scale = 10 }
    prior.y = { name = "uniform", args.loc = -5, args.scale = 10 }
    prior.z = { name = "uniform", args.loc = -5, args.scale = 10 }
    observations.x.file = "lorenz63-x.ssv"
    observations.y.file = "lorenz63-y.ssv"
    observations.z.file = "lorenz63-z.ssv"
    summary.tables.forecasts.component = "pypfilt.summary.PredictiveCIs"
    summary.tables.forecasts.credible_intervals = [50, 60, 70, 80, 90, 95]

    [scenario.forecast_regularised]
    prior.x = { name = "uniform", args.loc = -5, args.scale = 10 }
    prior.y = { name = "uniform", args.loc = -5, args.scale = 10 }
    prior.z = { name = "uniform", args.loc = -5, args.scale = 10 }
    observations.x.file = "lorenz63-x.ssv"
    observations.y.file = "lorenz63-y.ssv"
    observations.z.file = "lorenz63-z.ssv"
    summary.tables.forecasts.component = "pypfilt.summary.PredictiveCIs"
    summary.tables.forecasts.credible_intervals = [50, 60, 70, 80, 90, 95]
    filter.regularisation.enabled = true
    filter.regularisation.bounds.x = { min = -50, max = 50 }
    filter.regularisation.bounds.y = { min = -50, max = 50 }
    filter.regularisation.bounds.z = {}
    """) + '\n'
