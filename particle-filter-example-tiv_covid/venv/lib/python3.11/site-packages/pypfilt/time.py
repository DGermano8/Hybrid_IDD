"""Simulation time scales and units."""

import abc
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import math
import numpy as np
from typing import Any
import warnings


@dataclass(frozen=True)
class TimeStep:
    """
    The definition of a time-step.

    :param start: The beginning of the time-step.
    :param end: The end of the time-step.
    :param dt: The time-step size.
    """
    start: Any
    end: Any
    dt: float

    def __post_init__(self):
        """
        These validation checks ensure that when iterating over the time-steps
        returned by various :class:`Time` methods, the caller cannot construct
        a :class:`TimeStep` instance whose start or end is itself a
        :class:`TimeStep`.
        """
        if isinstance(self.start, TimeStep):
            raise ValueError('TimeStep.start cannot be a TimeStep')
        if isinstance(self.end, TimeStep):
            raise ValueError('TimeStep.end cannot be a TimeStep')


class Time(abc.ABC):
    """
    The base class for simulation time scales, which defines the minimal set
    of methods that are required.
    """

    def __init__(self):
        """
        Return a time scale with no defined simulation period.
        """
        self._start = None
        self._end = None
        self.steps_per_unit = None

    def set_period(self, start, end, steps_per_unit):
        """
        Define the simulation period and time-step size.

        :param start: The start of the simulation period.
        :param end: The end of the simulation period.
        :param steps_per_unit: The number of time-steps per unit time.
        :type steps_per_unit: int
        :raises ValueError: if ``start`` and/or ``end`` cannot be parsed, or
            if ``steps_per_unit`` is not a positive integer.
        """
        try:
            start = self.parse(start)
        except ValueError:
            raise ValueError("Invalid start: {}".format(start))
        try:
            end = self.parse(end)
        except ValueError:
            raise ValueError("Invalid end: {}".format(end))
        if not isinstance(steps_per_unit, int) or steps_per_unit < 1:
            msg = "Invalid steps_per_unit: {}".format(steps_per_unit)
            raise ValueError(msg)
        self._start = start
        self._end = end
        self.steps_per_unit = steps_per_unit

    def parse(self, value):
        """
        Attempt to convert a value into a time.

        :raises ValueError: if the value could not be converted into a time.
        """
        if isinstance(value, str):
            return self.from_unicode(value)
        elif not self.is_instance(value):
            msg_fmt = 'Cannot parse time of type: "{}"'
            raise ValueError(msg_fmt.format(type(value)))

    @abc.abstractmethod
    def dtype(self, name):
        """
        Define the dtype for columns that store times.
        """
        pass

    @abc.abstractmethod
    def native_dtype(self):
        """
        Define the Python type used to represent times in NumPy arrays.
        """
        pass

    @abc.abstractmethod
    def is_instance(self, value):
        """
        Return whether ``value`` is an instance of the native time type.
        """
        pass

    @abc.abstractmethod
    def to_dtype(self, time):
        """
        Convert from time to a dtype value.
        """
        pass

    @abc.abstractmethod
    def from_dtype(self, dval):
        """
        Convert from a dtype value to time.
        """
        pass

    @abc.abstractmethod
    def to_unicode(self, time):
        """
        Convert from time to a Unicode string.

        This is used to define group names in HDF5 files, and for logging.
        """
        pass

    @abc.abstractmethod
    def from_unicode(self, val):
        """
        Convert from a Unicode string to time.

        This is used to parse group names in HDF5 files.
        """
        pass

    @abc.abstractmethod
    def column(self, name):
        """
        Return a tuple that can be used with :func:`~pypfilt.io.read_table` to
        convert a column into time values.
        """
        pass

    @abc.abstractmethod
    def steps(self):
        """
        Return a generator that yields tuples of time-step numbers and
        :class:`TimeStep` instances, which span the simulation period.

        The first time-step should be numbered 1 and occur at a time that is
        one time-step after the beginning of the simulation period.
        """
        pass

    @abc.abstractmethod
    def step_count(self):
        """
        Return the number of time-steps required for the simulation period.
        """
        pass

    @abc.abstractmethod
    def step_of(self, time):
        """
        Return the time-step number that corresponds to the specified time.
        """
        pass

    @abc.abstractmethod
    def add_scalar(self, time, scalar):
        """
        Add a scalar quantity to the specified time.
        """
        pass

    @abc.abstractmethod
    def time_of_obs(self, obs):
        """
        Return the time associated with an observation.
        """
        pass

    def to_scalar(self, time):
        """
        Convert the specified time into a scalar quantity, defined as the
        time-step number divided by the number of time-steps per unit time.
        """
        return self.step_of(time) / self.steps_per_unit

    def time_step_of_next_observation(self, ctx, obs_tables, from_time):
        """
        Identify the first time-step *after* ``from_time`` for which there is
        at least one observation.
        This is returned as a tuple that contains the time-step number, the
        time-step details (:class:`TimeStep`), and a list of observations.
        If no such time-step exists, ``None`` is returned.

        :param ctx: The simulation context.
        :type ctx: pypfilt.Context
        :param obs_tables: The observation data tables.
        :type obs_tables: Dict[str, numpy.ndarray]
        :param from_time: The starting time (set to ``None`` to use the start
            of the simulation period).
        :raises ValueError: If observations are not sorted chronologically.
        """
        steps = self.with_observation_tables(ctx, obs_tables, start=from_time)
        for (step_num, time_step, obs_list) in steps:
            if obs_list:
                return (step_num, time_step, obs_list)

        return None

    def with_observation_tables(self, ctx, obs_tables, start=None):
        """
        Return a generator that yields a sequence of tuples that contain: the
        time-step number, the time-step details (:class:`TimeStep`), and a
        list of observations.

        :param ctx: The simulation context.
        :type ctx: pypfilt.Context
        :param obs_tables: The observation data tables.
        :type obs_tables: Dict[str, numpy.ndarray]
        :param start: The starting time (set to ``None`` to use the start of
            the simulation period).
        :raises ValueError: If observations are not sorted chronologically.
        """
        if start is None:
            start = self._start

        # Ensure that observations are sorted chronologically.
        for (obs_unit, obs_data) in obs_tables.items():
            for ix in range(1, len(obs_data)):
                t0 = obs_data['time'][ix - 1]
                t1 = obs_data['time'][ix]
                if t1 < t0:
                    msg = '"{}" observations are not ordered chronologically'
                    raise ValueError(msg.format(obs_unit))

        # For each observations table, record the row index and observation
        # dictionary of the next observation.
        obs_streams = {}
        for (obs_unit, obs_data) in obs_tables.items():
            obs_model = ctx.component['obs'][obs_unit]
            (row_ix, obs) = find_first_observation_after(
                obs_model, obs_data, start)
            if row_ix is None or obs is None:
                continue
            obs_streams[obs_unit] = (row_ix, obs)

        for (step, time_step) in self.steps():
            # Skip to the first time-step *after* the starting time.
            if time_step.end <= start:
                continue

            # Find all observations that pertain to this time-step.
            current_obs = []
            for (obs_unit, (row_ix, obs)) in list(obs_streams.items()):
                obs_model = ctx.component['obs'][obs_unit]
                obs_data = obs_tables[obs_unit]

                # Find all relevant observations from this observation stream.
                obs_times = set()
                search_from = row_ix + 1
                while obs['time'] <= time_step.end:
                    current_obs.append(obs)
                    obs_times.add(obs['time'])

                    # Find the next observation, if any.
                    (row_ix, obs) = find_first_observation_after(
                        obs_model, obs_data, start, first_ix=search_from)
                    if row_ix is None or obs is None:
                        # If there are no more observations, stop tracking
                        # this observation stream.
                        del obs_streams[obs_unit]
                        break

                    search_from = row_ix + 1

                # Warn the user if there are multiple observations for this
                # time-step, where the observation times are not identical.
                # This indicates that smaller time-steps might be appropriate.
                #
                # We use warnings.warn() as per the guidelines table in the
                # "When to use logging" section of the Python logging
                # documentation: this issue is avoidable, and the user can
                # modify the data to eliminate this warning.
                if len(obs_times) > 1:
                    msg = (
                        '{} "{}" observations for different times at t = {};'
                        ' consider using smaller time-steps')
                    level = 2
                    warnings.warn(
                        msg.format(len(obs_times), obs_unit, time_step.end),
                        UserWarning, stacklevel=level)

                # Record the next observation from this observation stream.
                if obs is not None:
                    obs_streams[obs_unit] = (row_ix, obs)

            # Yield a (step_number, time_step, observations) tuple.
            yield (step, time_step, current_obs)


def find_first_observation_after(obs_model, obs_data, start, first_ix=0):
    """
    Return the row index and observation dictionary for the first observation
    in ``obs_data`` that occurs **after** the time ``start``.
    If no such observation exists, return ``(None, None)``.

    :param obs_model: The observation model for these observations.
    :type obs_model: pypfilt.Obs
    :param obs_data: The observations table.
    :type obs_data: numpy.ndarray
    :param start: The time after which the next observation should occur.
    :param first_ix: The row index from which to begin searching.
    :type first_ix: int
    """
    row_ixs = range(first_ix, len(obs_data))
    for ix in row_ixs:
        row = obs_data[ix]
        obs = obs_model.row_into_obs(row)

        if obs['time'] > start:
            return (ix, obs)

    return (None, None)


def step_subset_modulos(steps_per_unit, samples_per_unit):
    """
    Return an array of modulos that identify the time-steps at which to draw
    samples, in order to sample at the specified rate.

    :param steps_per_unit: The number of time-steps per unit time.
    :param samples_per_unit: The number of samples per unit time.
    :raises ValueError: If ``steps_per_unit`` is not a multiple of
        ``samples_per_unit``.
    """
    if samples_per_unit < 1 or samples_per_unit > steps_per_unit:
        msg = 'Cannot take {} samples out of {}'
        raise ValueError(msg.format(samples_per_unit, steps_per_unit))
    if steps_per_unit % samples_per_unit != 0:
        msg = 'Cannot take {} samples out of {}'
        raise ValueError(msg.format(samples_per_unit, steps_per_unit))
    every_nth = steps_per_unit // samples_per_unit
    step_mods = 1 + np.array(list(range(steps_per_unit)), dtype=int)
    step_mods[-1] = 0
    return step_mods[every_nth - 1::every_nth]


class Datetime(Time):
    """
    A ``datetime`` scale where the time unit is days.

    By default, times are converted to/from string using the format string
    ``'%Y-%m-%d %H:%M:%S'``, with a fallback format string ``'%Y-%m-%d'``
    used to read date-only values.
    You can override these format strings by defining the following settings:

    .. code-block:: toml

       [time]
       datetime_formats = ["%Y-%m-%d", "%d %m %Y"]

    The first value will be used when saving times, and subsequent values will
    be used as fallback options for reading times.
    """

    def __init__(self, settings=None):
        """
        :param settings: An optional dictionary of simulation settings.
        """
        super(Datetime, self).__init__()
        if settings is None:
            settings = {}
        self.format_strings = settings.get('time', {}).get(
            'datetime_formats',
            ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d'])
        if len(self.format_strings) == 0:
            raise ValueError('The time.datetime_formats setting is empty')

    @contextmanager
    def custom_format_strings(self, format_strings):
        """
        Temporarily override the datetime format strings within a ``with``
        statement.

        :param format_strings: The format strings used to convert datetime
            values to/from strings.

        :Examples:

        >>> from datetime import datetime
        >>> from pypfilt.time import Datetime
        >>> time_scale = Datetime()
        >>> time_scale.to_unicode(datetime(2022, 8, 31))
        '2022-08-31 00:00:00'
        >>> with time_scale.custom_format_strings(['%d %b %Y']):
        ...     time_scale.to_unicode(datetime(2022, 8, 31))
        '31 Aug 2022'
        """
        original_strings = self.format_strings
        try:
            self.format_strings = format_strings
            yield
        finally:
            self.format_strings = original_strings

    def dtype(self, name):
        """Define the dtype for columns that store times."""
        # NOTE: np.issubdtype doesn't consider string lengths.
        return (name, np.dtype('S20'))

    def native_dtype(self):
        """Define the Python type used to represent times in NumPy arrays."""
        return 'O'

    def is_instance(self, value):
        """Return whether ``value`` is an instance of the native time type."""
        return isinstance(value, datetime.datetime)

    def to_dtype(self, time):
        """Convert from time to a dtype value."""
        return time.strftime(self.format_strings[0]).encode('utf-8')

    def from_dtype(self, dval):
        """Convert from a dtype value to time."""
        if self.is_instance(dval):
            return dval
        elif isinstance(dval, datetime.date):
            # Treat date values as the start of the day.
            return datetime.datetime(dval.year, dval.month, dval.day)
        if isinstance(dval, bytes):
            dval = dval.decode('utf-8')
        for fmt in self.format_strings:
            try:
                return datetime.datetime.strptime(dval, fmt)
            except ValueError:
                pass
        raise ValueError('Invalid datetime value: {}'.format(dval))

    def to_unicode(self, time):
        """Convert from time to a Unicode string."""
        return time.strftime(self.format_strings[0])

    def from_unicode(self, val):
        """Convert from a Unicode string to time."""
        return self.from_dtype(val)

    def parse(self, value):
        """Attempt to convert a value into a time."""
        return self.from_dtype(value)

    def column(self, name):
        """
        Return a tuple that can be used with :func:`~pypfilt.io.read_table` to
        convert a column into time values.
        """
        return (name, 'O', self.from_dtype)

    def steps(self):
        """
        Return a generator that yields a sequence of time-step numbers and
        times (represented as tuples) that span the simulation period.

        The first time-step should be numbered 1 and occur at a time that is
        one time-step after the beginning of the simulation period.
        """
        dt = 1.0 / self.steps_per_unit
        delta = datetime.timedelta(days=dt)
        one_day = datetime.timedelta(days=1)

        step = 1
        start = self._start
        end = start + delta
        while end <= self._end:
            yield (step, TimeStep(start=start, end=end, dt=dt))
            # Proceed to the next time-step.
            step += 1
            curr_day = step // self.steps_per_unit
            curr_off = step % self.steps_per_unit
            start = end
            end = self._start + curr_day * one_day + curr_off * delta

    def step_count(self):
        """
        Return the number of time-steps required for the simulation period.
        """
        return self.step_of(self._end)

    def step_of(self, time):
        """
        Return the time-step number that corresponds to the specified time.

        - For ``date`` objects, the time-step number marks the **start of that
          day**.
        - For ``datetime`` objects, the time-step number is rounded to the
          **nearest** time-step.
        - For all other values, this will raise ``ValueError``.
        """
        if not isinstance(time, datetime.datetime):
            if isinstance(time, datetime.date):
                time = datetime.datetime(time.year, time.month, time.day)

        if isinstance(time, datetime.datetime):
            diff = time - self._start
            frac_diff = diff - datetime.timedelta(diff.days)
            frac_day = frac_diff.total_seconds()
            frac_day /= datetime.timedelta(days=1).total_seconds()
            day_steps = diff.days * self.steps_per_unit
            day_frac = frac_day * float(self.steps_per_unit)
            return day_steps + round(day_frac)
        else:
            raise ValueError("{}.step_of() does not understand {}".format(
                type(self).__name__, type(time)))

    def add_scalar(self, time, scalar):
        """Add a scalar quantity to the specified time."""
        return time + datetime.timedelta(scalar)

    def time_of_obs(self, obs):
        """
        Return the time associated with an observation.
        """
        d = obs['time']
        if isinstance(d, datetime.datetime):
            return d
        else:
            raise ValueError("{}.{}() does not understand {}".format(
                type(self).__name__, "time_of_obs", type(d)))


class Scalar(Time):
    """
    A dimensionless time scale.
    """

    def __init__(self, settings=None):
        """
        :param settings: An optional dictionary of simulation settings.
        """
        super(Scalar, self).__init__()
        self.np_dtype = np.float64

    def dtype(self, name):
        """Define the dtype for columns that store times."""
        return (name, self.np_dtype)

    def native_dtype(self):
        """Define the Python type used to represent times in NumPy arrays."""
        return self.np_dtype

    def is_instance(self, value):
        """Return whether ``value`` is an instance of the native time type."""
        return isinstance(value, (float, self.np_dtype))

    def to_dtype(self, time):
        """Convert from time to a dtype value."""
        return self.np_dtype(time)

    def from_dtype(self, dval):
        """Convert from a dtype value to time."""
        if isinstance(dval, np.ndarray):
            return dval.item()
        else:
            return self.np_dtype(dval)

    def to_unicode(self, time):
        """Convert from time to a Unicode string."""
        return str(time)

    def from_unicode(self, val):
        """Convert from a Unicode string to time."""
        return float(val)

    def parse(self, value):
        """Attempt to convert a value into a time."""
        return float(value)

    def column(self, name):
        """
        Return a tuple that can be used with :func:`~pypfilt.io.read_table` to
        convert a column into time values.
        """
        return (name, self.np_dtype)

    def steps(self):
        """
        Return a generator that yields a sequence of time-step numbers and
        times (represented as tuples) that span the simulation period.

        The first time-step should be numbered 1 and occur at a time that is
        one time-step after the beginning of the simulation period.
        """
        dt = 1 / self.steps_per_unit
        start = self._start
        end = self._start + dt
        step = 1
        while end <= self._end:
            yield (step, TimeStep(start=start, end=end, dt=dt))
            # Proceed to the next time-step.
            step += 1
            curr_units = step // self.steps_per_unit
            curr_rems = step % self.steps_per_unit
            start = end
            end = self._start + curr_units + curr_rems * dt

    def step_count(self):
        """
        Return the number of time-steps required for the simulation period.
        """
        return self.step_of(self._end)

    def step_of(self, time):
        """
        Return the time-step number that corresponds to the specified time.
        Fractional values are rounded to the **nearest** time-step.
        """
        frac, whole = math.modf(time - self._start)
        frac_steps = round(frac * self.steps_per_unit)
        whole = int(whole)
        return (whole * self.steps_per_unit) + frac_steps

    def add_scalar(self, time, scalar):
        """Add a scalar quantity to the specified time."""
        return time + scalar

    def time_of_obs(self, obs):
        """
        Return the time associated with an observation.
        """
        return obs['time']
