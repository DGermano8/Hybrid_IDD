"""
Constructs simulation contexts for scenario instances.
"""

from contextlib import contextmanager
import copy
import h5py
import importlib
import lhs
import logging
import numpy as np
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional

from .scenario import Instance, override_dict
from .io import read_table, read_lookup_table, Lookup
from .time import step_subset_modulos


class Context(NamedTuple):
    """
    A simulation context, which contains all of the components, data tables,
    and scenario settings that are required in order to run estimation passes
    and forecasts.

    :param scenario_id: The scenario identifier for this context.
    :type scenario_id: str
    :param descriptor: The identifier descriptor, which describes the
        observation model parameter values for this context.
    :type descriptor: str
    :param settings: The dictionary of simulation settings.
    :type settings: Dict[str, Any]
    :param source: The (optional) TOML input for this specification.
    :type source: Optional[str]
    :param data: The dictionary of data tables.
    :type data: Dict[str, Any]
    :param component: The dictionary of simulation components.
    :type component: Dict[str, Any]
    :param event_handler: The dictionary of event-handler functions.
    :type event_handler: Dict[str, list[Callable]]
    :param all_observations: All of the available observations in a single
        list.
    :type all_observations: List[dict]
    """
    scenario_id: str
    descriptor: str
    settings: Dict[str, Any]
    source: Optional[str]
    data: Dict[str, Any]
    component: Dict[str, Any]
    event_handler: Dict[str, Any]
    all_observations: List[dict]

    def get_setting(self, keys, default=None):
        """
        Return the setting associated with a sequence of keys, if all keys are
        present, otherwise return ``default``.

        :Examples:

        >>> from pypfilt.build import Context
        >>> ctx = Context(
        ...     scenario_id='fake_scenario',
        ...     descriptor='',
        ...     source=None,
        ...     settings={'a': 1, 'b': {'c': 2}},
        ...     data={},
        ...     component={},
        ...     event_handler={},
        ...     all_observations=[],
        ... )
        >>> ctx.get_setting(['a'])
        1
        >>> ctx.get_setting(['b', 'c'])
        2
        >>> ctx.get_setting(['b', 'd']) is None
        True
        >>> ctx.get_setting(['b', 'd'], default=42)
        42
        """
        return get_chained(self.settings, keys, default=default)

    @contextmanager
    def override_settings(self, overrides):
        """
        Temporarily override settings within a ``with`` statement.

        This uses :func:`~pypfilt.scenario.override_dict` to override the
        current settings.

        :param dict overrides: The overriding values.

        :Examples:

        >>> from pypfilt.build import Context
        >>> ctx = Context(
        ...     scenario_id='fake_scenario',
        ...     descriptor='',
        ...     source=None,
        ...     settings={'a': 1, 'b': {'c': 2, 'd': 3}},
        ...     data={},
        ...     component={},
        ...     event_handler={},
        ...     all_observations=[],
        ... )
        >>> ctx.get_setting(['b', 'c'])
        2
        >>> ctx.get_setting(['b', 'd'])
        3
        >>> ctx.get_setting(['b', 'e']) is None
        True
        >>> with ctx.override_settings({'b': {'c': 1, 'e': 2}}):
        ...     ctx.get_setting(['b', 'c'])
        ...     # NOTE: 'd' should retain its original value.
        ...     ctx.get_setting(['b', 'd'])
        ...     ctx.get_setting(['b', 'e'])
        1
        3
        2
        >>> ctx.get_setting(['b', 'c'])
        2
        >>> ctx.get_setting(['b', 'd'])
        3
        >>> ctx.get_setting(['b', 'e']) is None
        True
        """
        original = {k: copy.deepcopy(v) for (k, v) in self.settings.items()}
        override_dict(self.settings, overrides)
        try:
            yield self
        finally:
            self.settings.clear()
            self.settings.update(original)

    def __str__(self):
        fmt = 'Context(scenario_id="{}", descriptor="{}")'
        return fmt.format(self.scenario_id, self.descriptor)

    def __repr__(self):
        """
        The goal of ``__repr__`` is to produce *unambiguous* output, while the
        goal of ``__str__`` is to produce *readable* output.

        In this case, these two methods can return the same output because the
        scenario ID and instance descriptor uniquely identify a specific
        instance of a specific scenario.
        """
        return str(self)

    def install_event_handler(self, event_name, handler_fn):
        """
        Register a function that should be called in response to an event.

        :param event_name: The event name.
        :param handler_fn: The event-handler function.
        """
        if event_name not in self.event_handler:
            self.event_handler[event_name] = [handler_fn]
        elif handler_fn not in self.event_handler[event_name]:
            self.event_handler[event_name].append(handler_fn)

    def call_event_handlers(self, event_name, *args, **kwargs):
        """
        Call all event-handler functions associated with an event name.

        :param event_name: The event name.
        :param \\*args: Positional arguments to pass to the event handlers.
        :param \\**kwargs: Keyword arguments to pass to the event handlers.

        :Examples:

        >>> from pypfilt.build import Context
        >>> ctx = Context(
        ...     scenario_id='fake_scenario',
        ...     descriptor='',
        ...     source=None,
        ...     settings={},
        ...     data={},
        ...     component={},
        ...     event_handler={},
        ...     all_observations=[],
        ... )
        >>> event_name = 'my_event'
        >>> def my_handler(message):
        ...     print(message)
        >>> ctx.install_event_handler(event_name, my_handler)
        >>> ctx.call_event_handlers(event_name, 'Hello world')
        Hello world
        """
        if event_name in self.event_handler:
            for handler_fn in self.event_handler[event_name]:
                handler_fn(*args, **kwargs)

    def prior_table(self):
        """
        Return the prior distribution table for parameters whose values are
        not contained in external data files.

        :Examples:

        .. code:: python

           def draw_new_prior_samples(ctx):
               cfg = settings['sampler']
               prng = ctx.component['random']['prior']
               particles = ctx.settings['filter']['particles']
               prior_table = ctx.prior_table()
               external_samples = ctx.external_prior_samples()
               return ctx.component['sampler'].draw_samples(
                   cfg, prng, particles, prior_table, external_samples)
        """
        return get_prior_table(self.settings)

    def external_prior_samples(self):
        """
        Return samples from the prior distribution for parameters whose values
        are contained in external data files.

        :Examples:

        .. code:: python

           def draw_new_prior_samples(ctx):
               cfg = settings['sampler']
               prng = ctx.component['random']['prior']
               particles = ctx.settings['filter']['particles']
               prior_table = ctx.prior_table()
               external_samples = ctx.external_prior_samples()
               return ctx.component['sampler'].draw_samples(
                   cfg, prng, particles, prior_table, external_samples)
        """
        return load_external_prior_samples(self.settings)

    def particle_count(self):
        """Return the number of particles in the ensemble."""
        return self.settings['filter']['particles']

    def start_time(self):
        """
        Return the start of the simulation period.

        If the simulation period has not been defined, this returns ``None``.
        """
        return self.component['time']._start

    def end_time(self):
        """
        Return the end of the simulation period.

        If the simulation period has not been defined, this returns ``None``.
        """
        return self.component['time']._end

    def time_steps(self):
        """
        Return a generator that yields a sequence of time-step numbers and
        times (represented as tuples) that span the simulation period.

        The first time-step is assigned the number 1 and occurs one time-step
        after the beginning of the simulation period.
        """
        return self.component['time'].steps()

    def time_units(self):
        """
        Return a generator that yields a sequence of time-step numbers and
        times (represented as tuples) that span the simulation period.

        This sequence does not include the start of the simulation period.
        """
        steps_per_unit = self.settings['time']['steps_per_unit']
        for (step_num, time_step) in self.time_steps():
            if step_num % steps_per_unit == 0:
                yield (step_num, time_step.end)

    def summary_times(self):
        """
        Return a generator that yields the sequence of time-step numbers and
        times (represented as tuples) for which summary statistics will be
        calculated.

        This sequence includes the start of the simulation period, which is
        assigned the time-step number zero.
        """
        steps_per_unit = self.settings['time']['steps_per_unit']
        summary_freq = self.settings['time']['summaries_per_unit']
        modulos = step_subset_modulos(steps_per_unit, summary_freq)

        # Start at the initial state (time-step zero).
        yield (0, self.settings['time']['sim_start'])

        for (step_num, time_step) in self.time_steps():
            if step_num % steps_per_unit in modulos:
                yield (step_num, time_step.end)

    def summary_count(self):
        """
        Return the number of time-steps for which summary statistics will be
        calculated.
        """
        return len(list(self.summary_times()))


class ChainedDict(dict):
    def get_chained(self, keys, default=None):
        return get_chained(self, keys, default=default)

    def set_chained(self, keys, value):
        return set_chained(self, keys, value)

    def set_chained_default(self, keys, value):
        return set_chained_default(self, keys, value)


def build_context(inst: Instance, obs_tables=None):
    """
    Return a simulation context for the provided scenario instance.

    :param inst: The scenario instance.
    :type inst: Instance
    :param obs_tables: The (optional) dictionary of observation tables; when
        not provided, these will be constructed from each observation file.

    :rtype: Context
    """
    scenario_id = inst.scenario_id
    descriptor = inst.descriptor

    # NOTE: we will apply missing defaults and add derived settings to this
    # dictionary, and we want to avoid modifying the original instance.
    settings = ChainedDict(copy.deepcopy(inst.settings))
    source = inst.source

    data = {}
    component = {}
    event_handler = {}

    ctx = Context(
        scenario_id=scenario_id,
        descriptor=descriptor,
        settings=settings,
        source=source,
        data=data,
        component=component,
        event_handler=event_handler,
        all_observations=[],
    )

    # Apply default values for settings that are not defined.
    apply_missing_defaults(settings)

    # Ensure that all required settings are defined.
    validate_required_settings(settings)

    # Apply default values that depend on other settings.
    apply_derived_defaults(settings)

    # Validate the ensemble partitions, if defined.
    validate_ensemble_partitions(settings)

    # Define parameters that are determined by existing parameters.
    define_fixed_parameters(settings)

    # Create and validate all of the simulation components.
    component['time'] = build_time(settings)
    validate_time(component['time'], settings)
    component['model'] = build_model(settings)
    component['sampler'] = build_sampler(settings)
    component['random'] = build_prngs(settings)

    # Draw samples from the prior distribution.
    prior_table = get_prior_table(settings)
    external_samples = load_external_prior_samples(settings)
    data['prior'] = component['sampler'].draw_samples(
        settings['sampler'],
        component['random']['prior'],
        settings['filter']['particles'],
        prior_table, external_samples)
    # Verify that all sampled values are finite.
    for (name, values)in data['prior'].items():
        if not np.all(np.isfinite(values)):
            msg = 'Non-finite prior samples for "{}"'
            raise ValueError(msg.format(name))

    # Create lookup tables.
    component['lookup'] = {}
    data['lookup'] = {}
    lookup_tables = build_lookup_tables(settings, component['time'])
    for (name, (lookup_data, lookup_table)) in lookup_tables.items():
        component['lookup'][name] = lookup_table
        data['lookup'][name] = lookup_data

    # Store lookup indices for each sample lookup table.
    data['lookup_ixs'] = sample_lookup_columns(
        ctx.settings, ctx.component['random']['hist_lookup_cols'],
        component['lookup'])

    # Create observation models and load observations.
    component['obs'] = build_observation_models(settings)
    if obs_tables is None:
        obs_tables = load_observations(
            settings, component['obs'], component['time'])
    data['obs'] = obs_tables

    # Collect all of the observations into a single flat list.
    for (obs_unit, obs_table) in data['obs'].items():
        obs_model = component['obs'][obs_unit]
        for row in obs_table:
            ctx.all_observations.append(obs_model.row_into_obs(row))

    # Create summary monitors and tables.
    component['summary_monitor'] = build_summary_monitors(settings)
    component['summary_table'] = build_summary_tables(settings)

    # Create the summary component.
    summary_name = get_chained(settings, ['components', 'summary'])
    kwargs = get_chained(settings, ['summary', 'init'], {})
    component['summary'] = instantiate(summary_name, ctx, **kwargs)

    return ctx


def get_chained(table, keys, default=None):
    """
    Return the value associated with a sequence of keys, if all keys are
    present, otherwise return ``default``.

    :Examples:

    >>> from pypfilt.build import get_chained
    >>> data = {'a': 1, 'b': {'c': 2}}
    >>> get_chained(data, ['a'])
    1
    >>> get_chained(data, ['b', 'c'])
    2
    >>> get_chained(data, ['b', 'd']) is None
    True
    >>> get_chained(data, ['b', 'd'], default=42)
    42
    """
    value = table

    for key in keys:
        # NOTE: if we encounter a non-dictionary type, return the default.
        if not isinstance(value, dict):
            return default
        try:
            value = value[key]
        except KeyError:
            return default

    return value


def set_chained(table, keys, value):
    """
    Create or update the value associated with a sequence of keys, creating
    missing keys as needed.

    :raises ValueError: if a key exists but is not a dictionary.

    :Examples:

    >>> from pypfilt.build import get_chained, set_chained
    >>> data = {'a': 1, 'b': {'c': 2}}
    >>> set_chained(data, ['b', 'c'], 3)
    >>> get_chained(data, ['b', 'c'])
    3
    >>> set_chained(data, ['x', 'y', 'z'], 'Hello')
    >>> get_chained(data, ['x', 'y', 'z'])
    'Hello'
    >>> try:
    ...     set_chained(data, ['b', 'c', 'd'], 'Invalid keys')
    ... except ValueError:
    ...     print('Cannot replace b.c with a dictionary')
    Cannot replace b.c with a dictionary
    >>> print(data)
    {'a': 1, 'b': {'c': 3}, 'x': {'y': {'z': 'Hello'}}}
    """
    keys = list(keys)
    last_ix = len(keys) - 1
    for (ix, key) in enumerate(keys):
        # Insert the value when we reach the final key.
        if last_ix == ix:
            table[key] = value
            return

        if key in table:
            if not isinstance(table[key], dict):
                msg_fmt = 'Cannot replace value at {}'
                raise ValueError(msg_fmt.format(keys[:ix + 1]))
        else:
            table[key] = {}

        table = table[key]


def set_chained_default(table, keys, value):
    """
    Insert a missing value associated with a sequence of keys.

    :Examples:

    >>> from pypfilt.build import get_chained, set_chained_default
    >>> data = {'a': 1, 'b': {'c': 2}}
    >>> set_chained_default(data, ['b', 'c'], 11)
    >>> get_chained(data, ['b', 'c'])
    2
    >>> set_chained_default(data, ['b', 'f', 'g'], 42)
    >>> get_chained(data, ['b', 'f', 'g'])
    42
    >>> try:
    ...     set_chained_default(data, ['b', 'c', 'd'], 'Invalid keys')
    ... except ValueError:
    ...     print('Cannot replace b.c with a dictionary')
    Cannot replace b.c with a dictionary
    >>> print(data)
    {'a': 1, 'b': {'c': 2, 'f': {'g': 42}}}
    """
    if get_chained(table, keys) is None:
        set_chained(table, keys, value)


def require_chained(table, keys):
    value = get_chained(table, keys)
    if value is None:
        msg_fmt = 'No value provided for {}'
        path_str = '.'.join(keys)
        raise ValueError(msg_fmt.format(path_str))
    return value


def apply_missing_defaults(settings):
    """
    Apply default values for settings that were not defined.
    """

    set_chained_default(
        settings,
        ['filter', 'reweight_or_fail'],
        False)
    set_chained_default(
        settings,
        ['filter', 'reweight', 'enabled'],
        True)
    set_chained_default(
        settings,
        ['filter', 'reweight', 'exponent'],
        1.0)
    set_chained_default(
        settings,
        ['filter', 'resample', 'enabled'],
        True)
    set_chained_default(
        settings,
        ['filter', 'resample', 'threshold'],
        0.25)
    set_chained_default(
        settings,
        ['filter', 'resample', 'method'],
        'deterministic')
    set_chained_default(
        settings,
        ['filter', 'resample', 'before_forecasting'],
        False)
    set_chained_default(
        settings,
        ['filter', 'regularisation', 'enabled'],
        False)
    set_chained_default(
        settings,
        ['filter', 'regularisation', 'bounds'],
        {})
    set_chained_default(
        settings,
        ['filter', 'regularisation', 'regularise_or_fail'],
        False)
    set_chained_default(
        settings,
        ['filter', 'regularisation', 'tolerance'],
        1e-8)
    set_chained_default(
        settings,
        ['filter', 'adaptive_fit', 'enabled'],
        False)
    set_chained_default(
        settings,
        ['filter', 'adaptive_fit', 'target_effective_fraction'],
        0.5)
    set_chained_default(
        settings,
        ['filter', 'adaptive_fit', 'exponent_tolerance'],
        0.001)

    # By default, do not save the particle history matrix in the output file.
    set_chained_default(
        settings,
        ['summary', 'save_history'],
        False)

    # By default, record summary statistics once per unit time.
    set_chained_default(
        settings,
        ['time', 'summaries_per_unit'],
        1)

    # If no observation models are defined, ensure that an empty dictionary is
    # present so that there is an (empty) iterator for code that inspects each
    # observation model (such as certain summary tables).
    set_chained_default(
        settings,
        ['observations'],
        {})

    # Ensure an empty dictionary is present if there is no prior distribution.
    set_chained_default(
        settings,
        ['prior'],
        {})

    # Ensure an empty dictionary is present if there are no lookup tables.
    set_chained_default(
        settings,
        ['lookup_tables'],
        {})

    # Ensure an empty dictionary is present if there are no sampler settings.
    set_chained_default(
        settings,
        ['sampler'],
        {})


def apply_derived_defaults(settings):
    """
    Apply default values for settings that were not defined, where the default
    value depends on other settings.

    Note that this should be called **after** validating the required settings
    with :py:func:`validate_required_settings`.
    """
    set_chained_default(
        settings,
        ['filter', 'minimal_estimation_run'],
        True)

    set_chained_default(
        settings,
        ['summary', 'only_forecasts'],
        False)
    set_chained_default(
        settings,
        ['summary', 'metadata', 'packages'],
        [])

    set_chained_default(
        settings,
        ['files', 'input_directory'],
        '.')
    set_chained_default(
        settings,
        ['files', 'output_directory'],
        '.')
    # NOTE: consider using `tempfile.mkdtemp()`.
    set_chained_default(
        settings,
        ['files', 'temp_directory'],
        tempfile.gettempdir())
    set_chained_default(
        settings,
        ['files', 'delete_cache_file_before_forecast'],
        False
    )
    set_chained_default(
        settings,
        ['files', 'delete_cache_file_after_forecast'],
        False
    )


def validate_required_settings(settings):
    """
    Ensure that all required settings are defined.

    :raises ValueError: if a required setting not defined.
    """

    for comp in ['time', 'model', 'sampler', 'summary']:
        require_chained(settings, ['components', comp])

    require_chained(settings, ['filter', 'particles'])
    require_chained(settings, ['filter', 'prng_seed'])
    require_chained(settings, ['filter', 'history_window'])
    require_chained(settings, ['time', 'start'])
    require_chained(settings, ['time', 'until'])
    require_chained(settings, ['time', 'steps_per_unit'])
    require_chained(settings, ['time', 'summaries_per_unit'])


def validate_ensemble_partitions(settings):
    """
    Ensure that ensemble partitions are valid (if defined).

    This includes checking that the probability masses sum to unity, and that
    the numbers of particles sum to the ensemble size.
    If valid partitions are defined, this adds the following settings to each
    partition:

    - ``filter.partition.mask``: a Boolean mask array that identifies the
      particles included in this partition.
    - ``filter.partition.slice``: a slice object that can be used to select
      the particles included in this partition.
      The slice object will **always** have a defined ``start`` and ``stop``.
    - ``filter.partition.reservoir_ix``: the index of the reservoir for this
      partition (if any); only defined if such a reservoir exists.

    .. note:: Use the slice object for indexing into state vectors, because it
       will return a **view**.
       In contrast, using the mask array will return a **copy**.

    If no partitions are defined, the ensemble is treated as one partition.

    :raises ValueError: if the partitions are invalid.
    """
    logger = logging.getLogger(__name__)
    net_px = settings['filter']['particles']

    partitions = settings['filter'].get('partition')
    if partitions is None:
        # Define a single partition that covers the entire ensemble.
        settings['filter']['partition'] = [{
            'weight': 1.0,
            'particles': net_px,
            'mask': np.ones(net_px, dtype=np.bool_),
            'slice': slice(0, net_px),
            'reservoir': False,
        }]
        return

    weights = []
    particles = []

    # Record the defined and used reservoirs.
    used_reservoirs = set()
    known_reservoirs = set()

    start_ix = 0
    n_parts = len(partitions)

    for (ix, partition) in enumerate(partitions):
        for key in ['weight', 'particles']:
            if key not in partition:
                msg = 'Partition #{} has no field "{}"'.format(ix + 1, key)
                raise ValueError(msg)

        # Ensure the partition weight lies within the unit interval.
        weight = partition['weight']
        if weight < 0 or weight > 1:
            msg = 'Partition #{} has invalid weight {}'.format(ix + 1, weight)
            raise ValueError(msg)

        weights.append(partition['weight'])

        # Ensure the partition contains a valid number of particles.
        px = partition['particles']
        if px < 1 or px > net_px:
            msg = 'Partition #{} has invalid number of particles {}'.format(
                ix + 1, px)
            raise ValueError(msg)
        particles.append(px)

        mask = np.zeros(net_px, dtype=np.bool_)
        mask[start_ix:start_ix + px] = True
        partition['mask'] = mask
        partition['slice'] = slice(start_ix, start_ix + px)

        start_ix += px

        # Ensure that the 'reservoir' key is defined for all partitions.
        is_reservoir = partition.get('reservoir', False)
        partition['reservoir'] = is_reservoir

        if 'reservoir_partition' in partition:
            # Ensure this partition is not a reservoir.
            if is_reservoir:
                msg = 'Reservoir #{} cannot have a reservoir'
                raise ValueError(msg.format(ix + 1))

            # Ensure the reservoir partition is a reservoir.
            res_part = partition['reservoir_partition']
            if res_part < 1 or res_part > n_parts:
                msg = 'Partition #{} has invalid reservoir #{}'
                raise ValueError(msg.format(ix + 1, res_part))

            res_ix = res_part - 1
            partition['reservoir_ix'] = res_ix
            if not partitions[res_ix].get('reservoir', False):
                msg = 'Partition #{} has invalid reservoir #{}'
                raise ValueError(msg.format(ix + 1, res_part))

            # Ensure the reservoir particle fraction is defined.
            if 'reservoir_fraction' not in partition:
                msg = 'Partition #{} has no field "{}"'
                raise ValueError(msg.format(ix + 1, 'reservoir_fraction'))

            # Ensure the reservoir particle fraction is valid.
            frac = partition['reservoir_fraction']
            if not (frac > 0.0 and frac < 1.0):
                msg = 'Partition #{} has invalid reservoir fraction {}'
                raise ValueError(msg.format(ix + 1, frac))

            # Check whether reservoir_fraction is large enough.
            num_sampled = round(px * partition['reservoir_fraction'])
            if num_sampled > 0:
                used_reservoirs.add(res_ix)
            else:
                msg = 'Partition #{} samples no particles from its reservoir'
                logger.warning(msg.format(ix + 1))

        # Handle reservoir partitions.
        if is_reservoir:
            known_reservoirs.add(ix)

    unused_reservoirs = known_reservoirs - used_reservoirs
    if unused_reservoirs:
        for res_ix in unused_reservoirs:
            # An unused reservoir with zero weight has no purpose.
            if partitions[res_ix]['weight'] == 0:
                logger.warning('Unused reservoir #{}'.format(res_ix + 1))

    if start_ix != net_px:
        msg = 'Partition particles sum to {}, expected {}'.format(
            start_ix, net_px)
        raise ValueError(msg)

    net_weight = np.sum(weights)
    if not np.allclose(net_weight, 1.0):
        msg = 'Partition weights sum to {}, expected 1.0'.format(net_weight)
        raise ValueError(msg)


def build_time(settings):
    """
    Validate the simulation period and return the simulation time scale.
    """
    # NOTE: this is called by Instance.time_scale(), in which case we pass
    # the instance settings to the time scale constructor. This means that
    # we cannot rely on setting defaults being applied, or any setting
    # validation.
    time_scale = instantiate(settings['components']['time'], settings)
    return time_scale


def validate_time(time_scale, settings):
    """
    Validate the simulation period, converting time values as appropriate.
    """
    try:
        time_start = time_scale.parse(settings['time']['start'])
    except ValueError:
        msg_fmt = 'Invalid value for time.start: "{}"'
        raise ValueError(msg_fmt.format(type(settings['time']['start'])))

    try:
        time_until = time_scale.parse(settings['time']['until'])
    except ValueError:
        msg_fmt = 'Invalid value for time.until: "{}"'
        raise ValueError(msg_fmt.format(type(settings['time']['until'])))

    # NOTE: we also record the start time as the epoch.
    settings['time']['epoch'] = time_start
    settings['time']['start'] = time_start
    settings['time']['until'] = time_until


def define_fixed_parameters(settings):
    """
    Define parameter values that are determined by existing parameters.
    """
    settings['time']['dt'] = 1.0 / settings['time']['steps_per_unit']


def build_model(settings):
    """
    Return the simulation model.
    """
    model = instantiate(settings['components']['model'])
    return model


def build_sampler(settings):
    """
    Return the prior distribution sampler.
    """
    sampler = instantiate(settings['components']['sampler'])
    return sampler


def build_prngs(settings):
    """
    Return the pseudo-random number generators.
    """
    seed = require_chained(settings, ['filter', 'prng_seed'])
    prng_names = ['resample', 'prior', 'model', 'hist_lookup_cols']
    return {name: np.random.default_rng(seed) for name in prng_names}


def get_prior_table(settings):
    """
    Return the prior distribution table for parameters whose values are not
    contained in external data files.
    """
    prior_table = {
        name: config for (name, config) in settings['prior'].items()
        if not config.get('external', False)
    }
    return prior_table


def get_external_prior_table(settings):
    """
    Return the prior distribution table for parameters whose values are
    contained in external data files.
    """
    prior_table = {
        name: config for (name, config) in settings['prior'].items()
        if config.get('external', False)
    }
    return prior_table


def load_external_prior_samples(settings):
    """
    Return samples from the prior distribution for parameters whose values are
    contained in external data files.
    """
    external_prior = get_external_prior_table(settings)
    external_samples = {}

    for (name, config) in external_prior.items():
        if 'table' in config and 'column' in config:
            # Read samples from a plain-text file.
            filename = os.path.join(
                settings['files']['input_directory'],
                config['table'])
            column = config['column']
            table_data = read_table(filename, [(column, np.float_)])
            external_samples[name] = table_data[column]

        elif 'hdf5' in config and 'dataset' in config and 'column' in config:
            # Read samples from an HDF5 dataset.
            filename = os.path.join(
                settings['files']['input_directory'],
                config['hdf5'])
            dataset = config['dataset']
            column = config['column']
            with h5py.File(filename, 'r') as f:
                table_data = f[dataset][()]
                external_samples[name] = table_data[column]
            pass

        else:
            # Error: no idea how to read samples for this parameter.
            msg_fmt = 'Cannot read external samples for {}'
            raise ValueError(msg_fmt.format(name))

    return external_samples


def build_lookup_tables(settings, time_scale):
    """
    Return the lookup tables.
    """
    table_defns = settings.get('lookup_tables')
    if table_defns is None:
        return {}

    data_dir = get_chained(settings, ['files', 'input_directory'])
    data_path = pathlib.Path(data_dir)
    tables = {}

    for (table_name, file_name) in table_defns.items():
        if not isinstance(file_name, str):
            try:
                file_name = file_name['file']
            except KeyError:
                msg_fmt = 'Lookup table "{}" has no file name'
                raise ValueError(msg_fmt.format(table_name))
        file_path = data_path / file_name
        data = read_lookup_table(file_path, time_scale)
        table = Lookup(data)
        tables[table_name] = (data, table)

    return tables


def sample_lookup_columns(settings, prng, lookup_tables):
    """
    Return the sampled lookup indices for each sample lookup.
    """
    table_defns = settings.get('lookup_tables', {})
    sample_names = []
    for (table_name, file_name) in table_defns.items():
        try:
            do_sample = file_name.get('sample_values', False)
            if do_sample:
                sample_names.append(table_name)
        except AttributeError:
            pass

    particles = settings['filter']['particles']
    sampled_ixs = {}
    # NOTE: use lhs.draw() to ensure the sampled values are representative of
    # the possible values {0, 1, ..., N-1}.
    for name in sample_names:
        if name not in lookup_tables:
            msg_fmt = 'Unknown sample lookup table: {}'
            raise ValueError(msg_fmt.format(name))

        num_values = lookup_tables[name].value_count()
        dist = {name: {'name': 'randint',
                       'args': {'low': 0, 'high': num_values}}}
        samples = lhs.draw(prng, particles, dist)
        sampled_ixs[name] = samples[name]

    return sampled_ixs


def build_observation_models(settings):
    """
    Return the observation models.
    """
    obs_model_defns = settings.get('observations')
    if obs_model_defns is None:
        return {}

    obs_models = {}
    for (obs_unit, obs_settings) in obs_model_defns.items():
        # Ensure that an observation model is defined.
        om_name = obs_settings.get('model')
        if om_name is None:
            msg_fmt = 'No observation model defined for {}'
            raise ValueError(msg_fmt.format(obs_unit))

        # Instantiate the observation model.
        obs_model = instantiate(om_name, obs_unit, obs_settings)

        obs_models[obs_unit] = obs_model

    return obs_models


def load_observations(settings, obs_models, time_scale):
    """
    Return the observation data tables.
    """
    logger = logging.getLogger(__name__)

    data_dir = get_chained(settings, ['files', 'input_directory'])
    data_path = pathlib.Path(data_dir)

    obs_tables = {}

    for (obs_unit, obs_model) in obs_models.items():
        obs_settings = get_chained(settings, ['observations', obs_unit])
        # Skip observation models that have no associated data file.
        if 'file' not in obs_settings:
            msg_fmt = 'No data file for {} observations'
            logger.debug(msg_fmt.format(obs_unit))
            continue

        # Load the observations.
        obs_file = data_path / obs_settings['file']
        kwargs = obs_settings.get('file_args', {})
        obs_table = obs_model.from_file(obs_file, time_scale, **kwargs)
        # Ensure the observation table includes a 'time' column.
        # Otherwise the simulation cache will not work as intended.
        if 'time' not in obs_table.dtype.names:
            msg_fmt = 'No "time" column in {} table at {}'
            raise ValueError(msg_fmt.format(obs_unit, obs_settings['file']))

        obs_tables[obs_unit] = obs_table

    return obs_tables


def build_summary_monitors(settings):
    """
    Return the summary monitors.
    """
    logger = logging.getLogger(__name__)

    monitor_defns = get_chained(settings, ['summary', 'monitors'])
    if monitor_defns is None:
        return {}

    monitors = {}
    for (name, monitor_settings) in monitor_defns.items():
        # Ensure that a monitor is defined.
        monitor_name = monitor_settings.get('component')
        if monitor_name is None:
            # NOTE: allow settings to be defined for monitors that are created
            # by custom summary components, but print a warning.
            msg_fmt = 'No component defined for monitor {}'
            logger.warning(msg_fmt.format(name))
            continue

        monitor_args = monitor_settings.get('init', {})
        monitor = instantiate(monitor_name, **monitor_args)
        monitors[name] = monitor

    return monitors


def build_summary_tables(settings):
    """
    Return the summary tables.
    """
    logger = logging.getLogger(__name__)

    table_defns = get_chained(settings, ['summary', 'tables'])
    if table_defns is None:
        return {}

    tables = {}
    for (name, table_settings) in table_defns.items():
        # Ensure that a table is defined.
        table_name = table_settings.get('component')
        if table_name is None:
            # NOTE: allow settings to be defined for tables that are created
            # by custom summary components, but print a warning.
            msg_fmt = 'No component defined for table {}'
            logger.warning(msg_fmt.format(name))
            continue

        table_args = table_settings.get('init', {})
        table = instantiate(table_name, **table_args)
        tables[name] = table

    return tables


def lookup(full_name):
    """
    Retrieve an object from a fully-qualified name.

    :param str full_name: The fully-qualified name.

    :Examples:

    >>> summary_fn = lookup('pypfilt.summary.HDF5')
    """
    last_dot = full_name.rfind('.')
    if last_dot < 0:
        raise ValueError('No module name in "{}"'.format(full_name))
    module_name = full_name[:last_dot]
    value_name = full_name[last_dot + 1:]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger = logging.getLogger(__name__)
        logger.debug('Could not import "{}"'.format(module_name))
        logger.debug('Trying again in the working directory ...')
        cwd = os.getcwd()
        sys.path.append(cwd)
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.path.pop()
        logger.debug('Successfully imported "{}"'.format(module_name))

    # This raises AttributeError if the attribute does not exist.
    value = getattr(module, value_name)
    return value


def instantiate(full_name, *args, **kwargs):
    """
    Instantiate an object from a class name.

    :param str full_name: The fully-qualified class name.
    :param \\*args: Positional constructor arguments (optional).
    :param \\**kwargs: Named constructor arguments (optional).

    :Examples:

    >>> time = instantiate('pypfilt.Datetime', settings={})
    """
    object_class = lookup(full_name)
    if not callable(object_class):
        raise ValueError('The value "{}" is not callable'.format(full_name))
    try:
        return object_class(*args, **kwargs)
    except TypeError:
        logger = logging.getLogger(__name__)
        logger.error('Attempted to call "{}" with arguments:'
                     .format(full_name))
        for arg in args:
            print('    {}'.format(arg))
        for (name, value) in kwargs.items():
            print('    {} = {}'.format(name, value))
        raise
