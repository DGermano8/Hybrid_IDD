"""Observation models: expected values and log likelihoods."""

import abc
import numpy as np
import scipy.stats

from . import io


def expect(ctx, snapshot, unit):
    """
    Return the expected observation value :math:`\\mathbb{E}[y_t]` for every
    every particle :math:`x_t` at time :math:`t`.

    :param ctx: The simulation context.
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param unit: The observation unit.
    :type unit: str
    :param hist: The particle history matrix.
    :param hist_ix: The index of time :math:`t` in the history matrix.
    """
    if unit in ctx.settings['observations']:
        obs_model = ctx.component['obs'][unit]
        return obs_model.expect(ctx, snapshot)
    else:
        raise ValueError("Unknown observation type '{}'".format(unit))


def log_llhd(ctx, snapshot, obs):
    """
    Return the log-likelihood :math:`\\mathcal{l}(y_t \\mid x_t)` for the
    observation :math:`y_t` and every particle :math:`x_t`.

    :param ctx: The simulation context.
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param obs: The list of observations for the current time-step.
    """
    log_llhd = np.zeros(snapshot.vec.shape)

    for o in obs:
        unit = o['unit']

        if unit not in ctx.settings['observations']:
            raise ValueError("Unknown observation type '{}'".format(unit))

        if unit not in ctx.component['obs']:
            raise ValueError('No observation model for "{}"'.format(unit))

        obs_model = ctx.component['obs'][unit]
        obs_llhd = obs_model.log_llhd(ctx, snapshot, o)
        log_llhd += obs_llhd
        ctx.call_event_handlers('log_llhd', o, obs_llhd, snapshot.weights)

    return log_llhd


def simulate(ctx, snapshot, unit, rng):
    """
    Return a random sample of :math:`y_t` for each particle :math:`x_t`.

    :param ctx: The simulation context.
    :param snapshot: The current particle states.
    :type snapshot: ~pypfilt.state.Snapshot
    :param unit: The observation unit.
    :param rng: The random number generator to use.
    """
    if unit in ctx.settings['observations']:
        obs_model = ctx.component['obs'][unit]
        return obs_model.simulate(ctx, snapshot, rng)
    else:
        raise ValueError("Unknown observation type '{}'".format(unit))


class Obs(abc.ABC):
    """
    The base class of observation models, which defines the minimal set of
    methods that are required.

    .. note:: The observation model constructor (``__init__``) must accept two
       positional arguments: the observation unit (string) and the observation
       model settings (dictionary).
    """

    @abc.abstractmethod
    def log_llhd(self, ctx, snapshot, obs):
        """
        Return the log-likelihood :math:`\\mathcal{l}(y_t \\mid x_t)` for the
        observation :math:`y_t` and every particle :math:`x_t`.

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        :param obs: An observation for the current time-step, :math:`y_t`.
        """
        pass

    @abc.abstractmethod
    def expect(self, ctx, snapshot):
        """
        Return the expected observation value :math:`\\mathbb{E}[y_t]` for
        every particle :math:`x_t`, at one or more times :math:`t`.

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        """
        pass

    @abc.abstractmethod
    def quantiles(self, ctx, snapshot, probs):
        r"""
        Return the values :math:`y_i` that satisfy:

        .. math::

           y_i = \inf\left\{ y : p_i \le
               \sum_i w_i \cdot \mathcal{L}(y_t \le y \mid x_t^i)\right\}

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        :param probs: The probabilities :math:`p_i`, which **must** be sorted
            in **ascending order**.
        """
        pass

    @abc.abstractmethod
    def llhd_in(self, ctx, snapshot, y0, y1):
        """
        Return the weighted likelihood that :math:`y_t \\in [y_0, y_1)`:

        .. math::

           \\sum_i w_i \\cdot \\mathcal{L}(y_0 \\le y_t < y_1 \\mid x_t^i)

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        :param y0: The (inclusive) minimum fraction of cases, :math:`y_0`.
        :param y1: The (exclusive) maximum fraction of cases, :math:`y_1`.
        """
        pass

    @abc.abstractmethod
    def simulate(self, ctx, snapshot, rng):
        """
        Return a random sample of :math:`y_t` for each particle :math:`x_t`.

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        :param rng: The random number generator to use.
        """
        pass

    @abc.abstractmethod
    def from_file(self, filename, time_scale):
        """
        Load observations from a space-delimited text file with column headers
        defined in the first line.

        :param filename: The file to read.
        :param time_scale: The simulation time scale.

        :return: The data table of observations.
        :rtype: numpy.ndarray

        .. note::

           Use :func:`~pypfilt.io.read_fields` to implement this method.
           See the example implementation, below.

        :Example:

        .. code-block:: python

           import numpy as np
           import pypfilt.io

           def from_file(self, filename, time_scale):
               fields = [pypfilt.io.time_field('time'), ('value', np.float_)]
               return pypfilt.io.read_fields(time_scale, filename, fields)
        """
        pass

    @abc.abstractmethod
    def row_into_obs(self, row):
        """
        Convert a data table row into an observation dictionary.

        :param row: The data table row.
        """
        pass

    @abc.abstractmethod
    def simulated_obs(self, ctx, snapshot, rng):
        """
        Return a simulated observation dictionary for each particle.

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        :param rng: The random number generator to use.
        """
        pass

    @abc.abstractmethod
    def simulated_field_types(self, ctx):
        """
        Return a list of ``(field_name, field_dtype, field_shape)`` tuples
        that define the vector for simulation observations.

        The third element, ``field_shape``, is optional and contains the shape
        of this field if it forms an array of type ``field_dtype``.

        .. note::

           Use :func:`pypfilt.io.time_field` for columns that will contain
           time values.
           This ensures that the time values will be converted as necessary
           when loading and saving tables.

        :param ctx: The simulation context.
        """
        pass

    @abc.abstractmethod
    def obs_into_row(self, obs, dtype):
        """
        Convert an observation dictionary into a data table row.

        :param obs: The observation dictionary.
        :param dtype: The NumPy dtype for the data table.
        """
        pass


def _is_discrete(dist):
    """
    Return ``True`` if ``dist`` describes a discrete random variable,
    otherwise ``False``.

    :raises ValueError: if ``dist`` is not a valid distribution.
    """
    # NOTE: retrieve the original distribution from a frozen distribution.
    rv_types = (scipy.stats.rv_discrete, scipy.stats.rv_continuous)
    if not isinstance(dist, rv_types):
        if hasattr(dist, 'dist'):
            dist = dist.dist
        else:
            raise ValueError('Invalid distribution {}'.format(dist))

    if isinstance(dist, scipy.stats.rv_discrete):
        return True
    elif isinstance(dist, scipy.stats.rv_continuous):
        return False
    else:
        raise ValueError('Invalid distribution {}'.format(dist))


def _is_continuous(dist):
    """
    Return ``True`` if ``dist`` describes a continuous random variable,
    otherwise ``False``.

    :raises ValueError: if ``dist`` is not a valid distribution.
    """
    return not _is_discrete(dist)


class Univariate(Obs):
    """
    Define observation models in terms of a univariate ``scipy.stats``
    distribution.

    Implement the :meth:`~Univariate.distribution` method and all of the
    :class:`Obs` methods will be automatically implemented.

    Each observation comprises a time (``'time'``) and a floating-point value
    (``'value'``).

    :param obs_unit: The observation unit, a unique identifier for the
        observations associated with this observation model.
    :param settings: The observation model settings dictionary.

    .. note:: Override the :meth:`~Univariate.log_llhd` method to handle,
       e.g., incomplete observations.

    .. note:: The observation unit is stored in the ``unit`` attribute, and
       the setting dictionary is stored in the ``settings`` attribute (see the
       example below).

    :Examples:

    >>> from pypfilt.obs import Univariate
    >>> # Define a Gaussian observation model with a known standard deviation.
    >>> class MyObsModel(Univariate):
    ...     def distribution(self, ctx, snapshot):
    ...         # Directly observe the state variable 'x'.
    ...         expect = snapshot.state_vec['x']
    ...         sdev = self.settings['parameters']['sdev']
    ...         return scipy.stats.norm(loc=expect, scale=self.sdev)
    ...
    >>> observation_unit = 'x'
    >>> settings = {'parameters': {'sdev': 0.1}}
    >>> obs_model = MyObsModel(observation_unit, settings)
    >>> obs_model.unit
    'x'

    The observation model shown in the example above can then be used in a
    scenario definition:

    .. code-block:: toml

       [observations.x]
       model = "my_module.MyObsModel"
       file = "x-observations.ssv"
       parameters.sdev = 0.2
    """

    def __init__(self, obs_unit, settings):
        self.unit = obs_unit
        self.settings = settings

    @abc.abstractmethod
    def distribution(self, ctx, snapshot):
        """
        Return a **frozen** ``scipy.stats`` distribution that defines the
        observation model for each particle.

        :param ctx: The simulation context.
        :param snapshot: The current particle states.
        :type snapshot: ~pypfilt.state.Snapshot
        """
        pass

    def quantiles_tolerance(self):
        """
        Return the minimum interval width when calculating quantiles for a
        continuous random variable.

        .. note:: The default tolerance is ``0.00001``.
           Override this method to adjust the tolerance.
        """
        return 0.00001

    def log_llhd(self, ctx, snapshot, obs):
        dist = self.distribution(ctx, snapshot)
        if _is_continuous(dist):
            return dist.logpdf(obs['value'])
        else:
            return dist.logpmf(obs['value'])

    def expect(self, ctx, snapshot):
        dist = self.distribution(ctx, snapshot)
        # NOTE: ensure we always return an array.
        return np.atleast_1d(dist.mean())

    def quantiles(self, ctx, snapshot, probs):
        dist = self.distribution(ctx, snapshot)

        # Ensure weights sum to unity, even if looking at a subset.
        weights = snapshot.weights / np.sum(snapshot.weights)

        def cdf(y):
            """Calculate the CDF of the weighted sum over all particles."""
            # NOTE: ignore division by zero and floating-point underflow
            # warnings, which can arise when there are outlier particles.
            # We subsequently check that the returned values are finite.
            with np.errstate(divide='ignore', under='ignore'):
                cdf_values = dist.cdf(y)
            if not np.all(np.isfinite(cdf_values)):
                raise ValueError('CDF has returned non-finite values')
            return np.dot(weights, cdf_values)

        # Determine the appropriate bisection method.
        if _is_continuous(dist):
            tolerance = self.quantiles_tolerance()

            def bisect(a, b):
                """
                Return the midpoint of the interval [a, b], or ``None`` if the
                minimum tolerance has been reached.
                """
                if b > a + tolerance:
                    return (a + b) / 2
                else:
                    return None

        else:
            def bisect(a, b):
                """
                Return the midpoint of the interval [a, b], or ``None`` if the
                minimum tolerance has been reached.
                """
                if b > a + 1:
                    return np.rint((a + b) / 2).astype(int)
                else:
                    return None

        # Find appropriate lower and upper bounds for y_i.
        pr_min = np.min(probs)
        pr_max = np.max(probs)

        with np.errstate(divide='ignore', under='ignore'):
            y0_lower = np.atleast_1d(dist.ppf(pr_min)).min()
            y0_upper = np.atleast_1d(dist.ppf(pr_max)).max()
        if not np.all(np.isfinite(y0_lower)):
            raise ValueError('PPF has returned non-finite lower values')
        if not np.all(np.isfinite(y0_upper)):
            raise ValueError('PPF has returned non-finite upper values')

        return bisect_cdf(probs, cdf, bisect, y0_lower, y0_upper)

    def llhd_in(self, ctx, snapshot, y0, y1):
        dist = self.distribution(ctx, snapshot)
        # NOTE: we integrate discrete variables from `y0` to `y1 - 1`.
        if _is_discrete(dist):
            y0 = y0 - 1
            y1 = y1 - 1
        # Ensure weights sum to unity, even if looking at a subset.
        weights = snapshot.weights / np.sum(snapshot.weights)
        return np.dot(weights, dist.cdf(y1) - dist.cdf(y0))

    def simulate(self, ctx, snapshot, rng):
        dist = self.distribution(ctx, snapshot)
        # NOTE: ensure we always return an array.
        return np.atleast_1d(dist.rvs(random_state=rng))

    def from_file(self, filename, time_scale,
                  time_col='time', value_col='value'):
        """
        Load count data from a space-delimited text file with column headers
        defined in the first line.

        :param filename: The file to read.
        :param time_scale: The simulation time scale.
        :param time_col: The name of the observation time column; this will be
            renamed to ``'time'``.
        :param value_col: The name of the observation value column; this will
            be renamed to ``'value'``.
        """
        # Load the data table.
        fields = [io.time_field(time_col), (value_col, np.float_)]
        df = io.read_fields(time_scale, filename, fields)
        # Rename the columns to 'time' and 'value'.
        rename_to = {
            time_col: 'time',
            value_col: 'value',
        }
        new_names = tuple(rename_to.get(name, name)
                          for name in df.dtype.names)
        df.dtype.names = new_names
        return df

    def row_into_obs(self, row):
        """
        Return an observation with fields ``'time'``, ``'value'``, and
        ``'unit'``.
        """
        return {
            'time': row['time'],
            'value': row['value'],
            'unit': self.unit,
        }

    def obs_into_row(self, obs, dtype):
        """
        Convert an observation into a ``(time, value)`` tuple.
        """
        return (obs['time'], obs['value'])

    def simulated_field_types(self, ctx):
        """
        Return the field types for simulated observations.
        """
        cols = [io.time_field('time'), ('value', np.float_)]
        return cols

    def simulated_obs(self, ctx, snapshot, rng):
        """
        Return a simulated observation with fields ``'time'`` and ``'value'``.
        """
        values = self.simulate(ctx, snapshot, rng)
        return [{'time': snapshot.time, 'value': value} for value in values]


def bisect_cdf(probs, cdf_fn, bisect_fn, y_lower, y_upper):
    r"""
    Use a bisection method to estimate the values :math:`y_i` that satisfy:

    .. math::

       y_i = \inf\left\{ y : p_i \le
           \sum_i w_i \cdot \mathcal{L}(y_t \le y \mid x_t^i)\right\}

    :param probs: The probabilities :math:`p_i`, which **must** be sorted in
        **ascending order**.
    :param cdf_fn: The CDF function
        :math:`f(y) = \sum_i w_i \cdot \mathcal{L}(y_t \le y \mid x_t^i)`.
    :param bisect_fn: The bisection function ``f(a, b)`` that either returns
        the midpoint of the interval :math:`[a, b]` or ``None`` if the
        search should stop (e.g., because a tolerance has been reached).
    :param y_lower: A lower bound for all :math:`y_i`.
    :param y_upper: An upper bound for all :math:`y_i`.
    """
    # NOTE: there may be definite lower/upper limits that cannot be exceeded
    # (e.g., 0 is a definite lower limit for Poisson and negative binomial
    # observation models). So we need to trust the observation models to
    # provide valid initial bounds, rather than checking them here.
    cdf_lower = cdf_fn(y_lower)

    bounds_lower = {pr: y_lower for pr in probs}
    bounds_upper = {pr: y_upper for pr in probs}
    qtls = np.zeros(probs.shape)
    for (ix, pr) in enumerate(probs):
        if cdf_lower >= pr:
            # The lower bound is the first value to meet or exceed this
            # threshold, so we've found y_i for this quantile.
            qtls[ix] = y_lower
            continue

        # Use a binary search to find y_i this quantile.
        y_lower = bounds_lower[pr]
        y_upper = bounds_upper[pr]

        y_mid = bisect_fn(y_lower, y_upper)
        while y_mid is not None:
            cdf_mid = cdf_fn(y_mid)

            # Check if this is a good lower or upper bound for any of the
            # remaining quantiles.
            for p in probs[ix+1:]:
                if cdf_mid <= p and y_mid > bounds_lower[p]:
                    bounds_lower[p] = y_mid
                if cdf_mid >= p and y_mid < bounds_upper[p]:
                    bounds_upper[p] = y_mid

            # Identify the half of the interval in which to search.
            if cdf_mid >= pr:
                y_upper = y_mid
            if cdf_mid <= pr:
                y_lower = y_mid
            y_mid = bisect_fn(y_lower, y_upper)

        # Record the value y_i for this quantile.
        qtls[ix] = y_upper

    return qtls
