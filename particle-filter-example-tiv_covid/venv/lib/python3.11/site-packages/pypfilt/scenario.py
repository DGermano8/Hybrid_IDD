"""
Provides a declarative means of defining estimation and forecasting scenarios.

The purpose of this module is to allow users to define and run simulations
**without writing any Python code**, and instead define all of the necessary
settings in `TOML`_ files.
"""

import copy
import itertools
import tomli
from typing import Any, Dict, NamedTuple, Optional


class Specification(NamedTuple):
    """
    A specification that defines any number of scenarios.

    :param global_settings: Default settings for all scenarios.
    :type global_settings: Dict[str, Any]
    :param scenario_settings: Settings specific to single scenarios.
        This is a dictionary that maps the setting ID to the settings that are
        specific to the identified scenario.
    :type scenario_settings: Dict[str, Any]
    :param source: The (optional) TOML input for this specification.
    :type source: Optional[str]
    """
    global_settings: Dict[str, Any]
    scenario_settings: Dict[str, Any]
    source: Optional[str]


class Scenario(NamedTuple):
    """
    The definition of a single scenario.

    :param scenario_id: The unique identifier for this scenario.
    :type scenario_id: str
    :param settings: The settings dictionary, which defines all
        of the simulation components and parameters.
    :type settings: Dict[str, Any]
    :param source: The (optional) TOML input for this specification.
    :type source: Optional[str]
    """
    scenario_id: str
    settings: Dict[str, Any]
    source: Optional[str]


class Instance(NamedTuple):
    """
    A single instance of a scenario.

    :param scenario_id: The scenario identifier for this instance.
    :type scenario_id: str
    :param settings: The settings dictionary, which defines all
        of the simulation components and parameters, including any that are
        specific to this instance.
    :type settings: Dict[str, Any]
    :param descriptor: The identifier descriptor, which describes the
        observation model parameter values for this specific instance.
    :type descriptor: str
    :param source: The (optional) TOML input for this specification.
    :type source: Optional[str]
    """
    scenario_id: str
    settings: Dict[str, Any]
    descriptor: str
    source: Optional[str]

    def __str__(self):
        fmt = 'Instance(scenario_id="{}", descriptor="{}")'
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

    def build_context(self, obs_tables=None):
        """
        Return a simulation context for this scenario instance.

        This simply calls :py:func:`pypfilt.build.build_context`.

        :param obs_tables: The (optional) dictionary of observation tables;
            when not provided, these will be constructed from each observation
            file.

        :rtype: pypfilt.build.Context
        """
        from . import build
        return build.build_context(self, obs_tables=obs_tables)

    def time_scale(self):
        """
        Return the time scale for this scenario instance.

        :rtype: pypfilt.time.Time
        """
        from .build import build_time
        return build_time(self.settings)


class ObsModelParams(NamedTuple):
    """
    Describes the parameter values for an observation model, and how to format
    the parameter names and values into an instance descriptor.

    :param unit: The observation unit, which is a unique identifier for this
        observation model and the observations to which it pertains.
    :type unit: str
    :param values: The parameter values for this observation model.
    :type values: Dict[str, Any]
    :param value_format: The format strings used to convert parameter values
        into strings.
    :type value_format: Dict[str, str]
    :param display_names: The strings used to represent each parameter in
        instance descriptors.
    :type display_names: Dict[str, str]
    """
    unit: str
    values: Dict[str, Any]
    value_format: Dict[str, str]
    display_names: Dict[str, str]


def load_instances(sources):
    """
    Iterate over scenario instances defined in one or more `TOML`_ sources.

    :param sources: A list of file-like objects and/or file paths.
        If ``sources`` is not a list, it will be treated as the only item of a
        list.

    :rtype: Iterator[Instance]

    :Examples:

    >>> import pypfilt
    >>> import pypfilt.build
    >>> import pypfilt.examples.predation
    >>> pypfilt.examples.predation.write_example_files()
    >>> forecast_times = [1.0, 3.0, 5.0, 7.0, 9.0]
    >>> config_file = 'predation.toml'
    >>> data_file = 'output.hdf5'
    >>> for instance in pypfilt.load_instances(config_file):
    ...     context = instance.build_context()
    ...     state = pypfilt.forecast(context, forecast_times,
    ...                              filename=data_file)
    >>> # Remove the output file when it is no longer needed.
    >>> import os
    >>> os.remove(data_file)
    """
    for spec in load_specifications(sources):
        for scenario in scenarios(spec):
            for instance in instances(scenario):
                # NOTE: this is where the job of this module ends,
                # and the job of Context begins.
                yield instance


def load_toml(source):
    """
    Read `TOML`_ content from ``source`` and return the parsed dictionary and
    the `TOML`_ input.

    :param source: A file-like object or a file path.
    :return: A ``(dict, str)`` tuple.
    """
    if hasattr(source, 'read'):
        toml_string = source.read()
    else:
        with open(source, encoding='utf-8') as f:
            toml_string = f.read()

    parsed_dict = tomli.loads(toml_string)
    return (parsed_dict, toml_string)


def load_specifications(sources):
    """
    Iterate over the scenario specifications in ``sources``.

    :param sources: A list of file-like objects and/or file paths.
        If ``sources`` is not a list, it will be treated as a list containing
        one item.

    :rtype: Iterator[Specification]

    :raises ValueError: if a source does not define any scenarios.
    """
    sources = as_list(sources)

    for source in sources:
        (source_dict, toml_string) = load_toml(source)

        if 'scenario' not in source_dict:
            raise ValueError('No scenarios defined in {}'.format(source))

        scenarios_table = source_dict['scenario']
        del source_dict['scenario']

        spec = Specification(
            global_settings=source_dict,
            scenario_settings=scenarios_table,
            source=toml_string,
        )
        yield spec


def scenarios(spec):
    """
    Iterate over the scenarios in the provided specification ``spec``.

    :param spec: The scenario specifications.
    :type spec: Specification

    :rtype: Iterator[Scenario]
    """
    for (scenario_id, scenario_dict) in spec.scenario_settings.items():
        # Construct the scenario settings by applying scenario-specific
        # settings on top of the global settings.
        global_dict = copy.deepcopy(spec.global_settings)
        scenario_dict = copy.deepcopy(scenario_dict)
        settings = override_dict(global_dict, scenario_dict)

        scenario = Scenario(
            scenario_id=scenario_id,
            settings=settings,
            source=spec.source,
        )
        yield scenario


def instances(scenario):
    """
    Iterate over the instances of a single scenario.

    :param scenario: The scenario definition.
    :type scenario: Scenario

    :rtype: Iterator[Instance]
    """
    # Iterate over every combination of observation model parameter values.
    previous_descriptors = set()
    obs_combs = scenario_observation_model_combinations(scenario)
    for (value_dicts, descriptor) in obs_combs:

        # First ensure that the descriptor is unique.
        if descriptor in previous_descriptors:
            msg_fmt = 'Scenario "{}" has a duplicate descriptor "{}"'
            raise ValueError(msg_fmt.format(scenario.scenario_id, descriptor))
        previous_descriptors.add(descriptor)

        # Copy the scenario settings, and apply the parameter values for each
        # observation model.
        settings = copy.deepcopy(scenario.settings)
        for (obs_unit, values) in value_dicts.items():
            settings['observations'][obs_unit]['parameters'] = values

        # Return this instance of the scenario.
        instance = Instance(
            scenario_id=scenario.scenario_id,
            settings=settings,
            descriptor=descriptor,
            source=scenario.source,
        )
        yield instance


def observation_model_parameter_combinations(obs_params):
    """
    Iterate over every combination of parameter values for a single
    observation model.

    Each combination is returned as a ``(unit, values, descriptor)`` tuple.

    :param obs_params: The observation model parameters definition.
    :type obs_params: ObsModelParams

    :rtype: Iterator[tuple[str, Dict[str, float | int], str]]
    """
    # NOTE: sort parameters by name to ensure a consistent ordering.
    names = sorted(obs_params.values.keys())

    # Create a format string for each parameter.
    if obs_params.value_format and obs_params.display_names:
        # For example, if the 'bg_obs' parameter has the display name 'bg',
        # the format string will be "bg-{val[0]:{fmt[bg_obs]}}".
        out_fields = []
        for (ix, name) in enumerate(names):
            # NOTE: produce format strings such as .
            field = '{0}-{{values[{1}]:{{formats[{2}]}}}}'.format(
                obs_params.display_names[name], ix, name)
            out_fields.append(field)

        # Join the format strings into a single format string for all
        # parameters.
        out_fmt = '-'.join(out_fields)
    else:
        # If a descriptor table was not provided, use an empty string.
        # This will cause `instances()` to raise a ValueError if the scenario
        # has more than one instance.
        out_fmt = ''

    # NOTE: the parameters must be scanned in their listed order, so that the
    # order of the values matches that of the indices in the format string.
    scan = [as_list(obs_params.values[name]) for name in names]
    for parameter_values in itertools.product(*scan):
        values_dict = dict(zip(names, parameter_values))
        descriptor = out_fmt.format(values=parameter_values,
                                    formats=obs_params.value_format)
        yield (obs_params.unit, values_dict, descriptor)


def as_list(values):
    """
    Return values as a list.

    :param values: A list of values, or a value that will be returned as the
        only item of the returned list.
    :type values: Union[list[Any], Any]

    :rtype: list[Any]
    """
    if isinstance(values, list):
        return values
    else:
        return [values]


def scenario_observation_model_combinations(scenario):
    """
    Iterate over every combination of parameter values for each observation
    model.

    Each combination is returned as a ``(values, descriptor)`` tuple, where
    ``values`` is a dictionary that maps each observation model (identified by
    observation unit) to the
    parameter values for that observation model.

    :rtype: Iterator[tuple[Dict[str, Any], str]]
    """
    # NOTE: if the scenario has no observation models, return an empty
    # configuration dictionary and an empty descriptor string.
    if 'observations' not in scenario.settings:
        yield ({}, "")
        return

    obs_models = scenario_observation_model_parameters(scenario)
    obs_model_values = [
        observation_model_parameter_combinations(obs_model)
        for obs_model in obs_models.values()
    ]
    for obs_model_comb in itertools.product(*obs_model_values):
        # NOTE: each element is (unit, values_dict, descriptor)
        descriptors = [descr for (_unit, _values, descr) in obs_model_comb
                       if descr]
        if descriptors:
            descriptor = '-'.join(descriptors)
        else:
            descriptor = ''
        obs_config = {
            unit: values
            for (unit, values, _descr) in obs_model_comb
        }
        yield(obs_config, descriptor)


def scenario_observation_model_parameters(scenario):
    """
    Return the parameter values for each observation model in a scenario,
    where each observation model is identified by its observation unit.

    :param scenario: The scenario definition.
    :type scenario: Scenario

    :rtype: Dict[str, ObsModelParams]

    :raises ValueError: if the parameter names are not consistent across the
        parameter values, the value format strings, and the parameter display
        names.
    """
    obs_tables = scenario.settings['observations'].items()
    # NOTE: allow the descriptor table to be omitted.
    obs_models = {
        unit: ObsModelParams(
            unit=unit,
            values=om_dict.get('parameters', {}),
            value_format=om_dict.get('descriptor', {}).get('format', {}),
            display_names=om_dict.get('descriptor', {}).get('name', {}),
        )
        for (unit, om_dict) in obs_tables
    }

    # Ensure that the parameter values, format string, and display names all
    # refer to the same set of parameters.
    for om_params in obs_models.values():
        value_keys = set(om_params.values.keys())
        format_keys = set(om_params.value_format.keys())
        names_keys = set(om_params.display_names.keys())
        identical_keys = (
            value_keys == format_keys
            and format_keys == names_keys
            and names_keys == value_keys)
        has_descriptor = format_keys or names_keys
        if not identical_keys and has_descriptor:
            msg_fmt = 'Invalid "{}" observation model'
            raise ValueError(msg_fmt.format(om_params.unit))

    return obs_models


def override_dict(defaults, overrides):
    """
    Override a dictionary with values in another dictionary. This will
    recursively descend into matching nested dictionaries.

    Where an override value is a dictionary, the corresponding default value
    must be a dictionary in order for nested defaults to be propagated.
    Otherwise, the default value is simply replaced by the override value.

    :param dict defaults: The original values; note that this dictionary
        **will be modified**.
    :param dict overrides: The overriding values.
    :return: The modified ``defaults`` dictionary.
    :rtype: Dict[Any, Any]
    """
    for (key, value) in overrides.items():
        if isinstance(value, dict):
            if key in defaults and isinstance(defaults[key], dict):
                # Override the nested default values.
                sub_defaults = defaults[key]
                defaults[key] = override_dict(sub_defaults, value)
            else:
                # Replace the default value with this dictionary.
                defaults[key] = value
        else:
            defaults[key] = value
    return defaults
