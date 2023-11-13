"""Calculation of simulation summary statistics."""

import abc
import h5py
import hashlib
import importlib
import locale
import logging
import numpy as np
import numpy.lib.recfunctions as nprec
import subprocess
import sys

from . import cache
from .io import fields_dtype, string_field, time_field, save_dataset
from . import resample
from . import state
from . import version


class Table(abc.ABC):
    """
    The base class for summary statistic tables.

    Tables are used to record rows of summary statistics as a simulation
    progresses.
    """

    @abc.abstractmethod
    def field_types(self, ctx, obs_list, name):
        """
        Return the column names and data types, represented as a list of
        ``(name, data type)`` tuples. See the NumPy documentation for details.

        .. note::

           To ensure that time and string values are handled appropriately
           when loading and saving tables, use :func:`pypfilt.io.time_field`
           to define time columns and :func:`pypfilt.io.string_field` to
           define string columns.
           For example:

           .. code-block:: python

              fields = [time_field('fs_time'), time_field('time'),
                        string_field('name), ('value', float)]

        :param ctx: The simulation context.
        :param obs_list: A list of all observations.
        :param name: The table's name.
        """
        pass

    @abc.abstractmethod
    def n_rows(self, ctx, forecasting):
        """
        Return the number of rows required for a single simulation.

        :param ctx: The simulation context.
        :param forecasting: ``True`` if this is a forecasting simulation,
            otherwise ``False``.
        """
        pass

    @abc.abstractmethod
    def add_rows(self, ctx, fs_time, window, insert_fn):
        """
        Record rows of summary statistics for some portion of a simulation.

        :param ctx: The simulation context.
        :param fs_time: The forecasting time; if this is not a forecasting
            simulation, this is the time at which the simulation ends.
        :param window: A list of :class:`~pypfilt.state.Snapshot` instances
            that capture the particle states at each summary time in the
            simulation window.
        :param insert_fn: A function that inserts one or more rows into the
            underlying data table; see the examples below.

        The row insertion function can be used as follows:

        .. code-block:: python

           # Insert a single row, represented as a tuple.
           insert_fn((x, y, z))
           # Insert multiple rows, represented as a list of tuples.
           insert_fn([(x0, y0, z0), (x1, y1, z1)], n=2)
        """
        pass

    def finished(self, ctx, fs_time, window, insert_fn):
        """
        Record rows of summary statistics at the end of a simulation.

        The parameters are as per :meth:`.add_rows`.

        Derived classes should only implement this method if rows must be
        recorded by this method; the provided method does nothing.
        """
        pass

    def load_state(self, ctx, grp):
        """
        Load the table state from a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object from which to load the state.
        """
        pass

    def save_state(self, ctx, grp):
        """
        Save the table state to a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object in which to save the state.
        """
        pass


class Monitor(abc.ABC):
    """
    The base class for simulation monitors.

    Monitors are used to calculate quantities that:

    * Are used by multiple Tables (i.e., avoiding repeated computation); or
    * Require a complete simulation for calculation (as distinct from Tables,
      which incrementally record rows as a simulation progresses).

    The quantities calculated by a Monitor can then be recorded by
    :meth:`.Table.add_rows` and/or :meth:`.Table.finished`.
    """

    def prepare(self, ctx, obs_list, name):
        """
        Perform any required preparation prior to a set of simulations.

        :param ctx: The simulation context.
        :param obs_list: A list of all observations.
        :param name: The monitor's name.
        """
        pass

    def begin_sim(self, ctx, forecasting):
        """
        Perform any required preparation at the start of a simulation.

        :param ctx: The simulation context.
        :param forecasting: ``True`` if this is a forecasting simulation,
            otherwise ``False``.
        """
        pass

    def monitor(self, ctx, fs_time, window):
        """
        Monitor the simulation progress.

        :param ctx: The simulation context.
        :param fs_time: The forecasting time; if this is not a forecasting
            simulation, this is the time at which the simulation ends.
        :param window: A list of :class:`~pypfilt.state.Snapshot` instances
            that capture the particle states at each summary time in the
            simulation window.
        """
        pass

    def end_sim(self, ctx, fs_time, window):
        """
        Finalise the data as required for the relevant summary statistics.

        The parameters are as per :meth:`.monitor`.

        Derived classes should only implement this method if finalisation of
        the monitored data is required; the provided method does nothing.
        """
        pass

    @abc.abstractmethod
    def load_state(self, ctx, grp):
        """
        Load the monitor state from a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object from which to load the state.
        """
        pass

    @abc.abstractmethod
    def save_state(self, ctx, grp):
        """
        Save the monitor state to a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object in which to save the state.
        """
        pass


class ParamCovar(Table):
    """
    Calculate the covariance between all pairs of model parameters during each
    simulation.
    """

    def field_types(self, ctx, obs_list, name):
        # Only calculate covariances between model parameters that admit
        # continuous kernels.
        model = ctx.component['model']
        # NOTE: model.can_smooth() returns a set, we need a list.
        self.__smooth_fields = [n for n in model.can_smooth()]
        self.__num_params = len(self.__smooth_fields)

        fs_time = time_field('fs_time')
        time = time_field('time')
        param1 = string_field('param1')
        param2 = string_field('param2')
        covar = ('covar', np.float64)
        return [fs_time, time, param1, param2, covar]

    def n_rows(self, ctx, forecasting):
        n_times = ctx.summary_count()
        return n_times * self.__num_params * (self.__num_params - 1) // 2

    def add_rows(self, ctx, fs_time, window, insert_fn):
        from . import stats

        for snapshot in window:
            x = state.repack(snapshot.state_vec[self.__smooth_fields])
            covars = stats.cov_wt(x, snapshot.weights)
            for ix1 in range(self.__num_params):
                name1 = self.__smooth_fields[ix1]
                for ix2 in range(ix1 + 1, self.__num_params):
                    name2 = self.__smooth_fields[ix2]
                    row = (fs_time, snapshot.time, name1, name2,
                           covars[ix1, ix2])
                    insert_fn(row)


class ModelCIs(Table):
    """
    Calculate fixed-probability central credible intervals for all state
    variables and model parameters.

    .. note:: Credible intervals are only recorded for **scalar** fields.
       Non-scalar fields will be ignored.

    The default intervals are: 0%, 50%, 60%, 70%, 80%, 90%, 95%, 99%, 100%.
    These can be overridden in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.tables]
       model_cints.component = "pypfilt.summary.ModelCIs"
       model_cints.credible_intervals = [ 0, 50, 95 ]
    """

    def __init__(self):
        self.__probs = np.uint8([0, 50, 60, 70, 80, 90, 95, 99, 100])

    def field_types(self, ctx, obs_list, name):
        # Identify the scalar fields.
        self.__field_names = [
            field[0] for field in ctx.component['model'].field_types(ctx)
            if len(field) == 2
        ]
        self.__stat_info = ctx.component['model'].stat_info()
        self.__num_stats = len(self.__field_names) + len(self.__stat_info)
        self.__probs = np.uint8(ctx.get_setting(
            ['summary', 'tables', name, 'credible_intervals'],
            self.__probs))

        fs_time = time_field('fs_time')
        time = time_field('time')
        prob = ('prob', np.int8)
        ymin = ('ymin', np.float64)
        ymax = ('ymax', np.float64)
        # State variables/parameters ('model') or statistics ('stat').
        value_type = string_field('type')
        name = string_field('name')
        return [fs_time, time, prob, ymin, ymax, value_type, name]

    def n_rows(self, ctx, forecasting):
        # Need a row for each interval, for each day, for each parameter,
        # variable and statistic.
        n_times = ctx.summary_count()
        return n_times * len(self.__probs) * self.__num_stats

    def add_rows(self, ctx, fs_time, window, insert_fn):
        from . import stats

        for snapshot in window:
            # Identify which state vectors to examine.
            valid = ctx.component['model'].is_valid(
                snapshot.state_vec)
            if valid.ndim != 1:
                raise ValueError('model.is_valid(): expected 1D array')
            if valid.shape != snapshot.weights.shape:
                raise ValueError('model.is_valid(): unexpected length')
            # Note that np.where() returns a *tuple*  of arrays (one for
            # each dimension) and we're only scanning a 1D array.
            mask = np.where(valid)[0]

            if np.count_nonzero(mask):
                ws = snapshot.weights[mask]
                for field in self.__field_names:
                    sub_hist = snapshot.state_vec[field][mask]
                    cred_ints = stats.cred_wt(sub_hist, ws, self.__probs)
                    for cix, pctl in enumerate(self.__probs):
                        row = (fs_time, snapshot.time, pctl,
                               cred_ints[pctl][0], cred_ints[pctl][1],
                               'model', field)
                        insert_fn(row)
                for (val, stat_fn) in self.__stat_info:
                    stat_vec = stat_fn(snapshot.state_vec[mask])
                    cred_ints = stats.cred_wt(stat_vec, ws, self.__probs)
                    for cix, pctl in enumerate(self.__probs):
                        row = (fs_time, snapshot.time, pctl,
                               cred_ints[pctl][0], cred_ints[pctl][1],
                               'stat', val)
                        insert_fn(row)
            else:
                for pctl in self.__probs:
                    for field in self.__field_names:
                        row = (fs_time, snapshot.time, pctl, 0, 0,
                               'model', field)
                        insert_fn(row)
                    for (val, _) in self.__stat_info:
                        row = (fs_time, snapshot.time, pctl, 0, 0, 'stat', val)
                        insert_fn(row)


class EnsembleSnapshot(Table):
    """
    Record the particle state vectors at each summary time of the estimation
    and forecasting passes.

    .. note:: These snapshots capture the ensemble at each summary time.
       There is **no relationship** between the particle ordering at different
       times.

    .. code-block:: toml

       [summary.tables]
       snapshot.component = "pypfilt.summary.EnsembleSnapshot"
    """

    def field_types(self, ctx, obs_list, name):
        # Create a PRNG for resampling the particles.
        prng_seed = ctx.settings['filter'].get('prng_seed')
        self.__resample = np.random.default_rng(prng_seed)
        fs_time = time_field('fs_time')
        time = time_field('time')
        weight = ('weight', np.float_)
        fields = ctx.component['model'].field_types(ctx)
        return [fs_time, time, weight] + fields

    def n_rows(self, ctx, forecasting):
        num_particles = ctx.particle_count()
        n_times = ctx.summary_count()
        return n_times * num_particles

    def add_rows(self, ctx, fs_time, window, insert_fn):
        for snapshot in window:
            # Record the state of each particle.
            for (row_ix, row) in enumerate(snapshot.state_vec):
                weight = snapshot.weights[row_ix]
                insert_fn(tuple((fs_time, snapshot.time, weight, *row)))


class ForecastSnapshot(Table):
    """
    Record the particle state vectors at the start of each forecasting pass
    and, optionally, at each day of the forecasting pass.

    .. note:: The particles will be resampled and so the state vectors will
       have uniform weights (which are not recorded in the table).

    .. code-block:: toml

       [summary.tables]
       snapshot.component = "pypfilt.summary.ForecastSnapshot"
       snapshot.each_summary_time = true
    """

    def field_types(self, ctx, obs_list, name):
        # Create a PRNG for resampling the particles.
        prng_seed = ctx.settings['filter'].get('prng_seed')
        self.__resample = np.random.default_rng(prng_seed)
        self.__each_summary_time = ctx.settings.get_chained(
            ['summary', 'tables', name, 'each_summary_time'],
            False)

        fs_time = time_field('fs_time')
        time = time_field('time')
        fields = ctx.component['model'].field_types(ctx)
        fields.insert(0, fs_time)
        fields.insert(1, time)
        return fields

    def n_rows(self, ctx, forecasting):
        if not forecasting:
            return 0

        if self.__each_summary_time:
            num_snapshots = ctx.summary_count()
        else:
            num_snapshots = 1
        # Produce one row for each particle.
        num_particles = ctx.particle_count()
        return num_particles * num_snapshots

    def add_rows(self, ctx, fs_time, window, insert_fn):
        for snapshot in window:
            if snapshot.time < fs_time:
                continue
            if not self.__each_summary_time:
                if snapshot.time != fs_time:
                    continue

            (sample_ixs, _weight) = resample.resample_weights(
                snapshot.weights, self.__resample)

            # Record the state of each particle.
            for ix in sample_ixs:
                row = snapshot.state_vec[ix]
                insert_fn(tuple((fs_time, snapshot.time, *row)))

    def load_state(self, ctx, group):
        """Restore the state of each PRNG from the cache."""
        cache.load_rng_states(group, 'prng_states', {
            'resample': self.__resample,
        })

    def save_state(self, ctx, group):
        """Save the current state of each PRNG to the cache."""
        cache.save_rng_states(group, 'prng_states', {
            'resample': self.__resample,
        })


class SimulatedObs(Table):
    """
    Record simulated observations for a single observation unit, for each
    particle in the simulation.

    The observation unit must be specified in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.tables]
       sim_obs.component = "pypfilt.summary.SimulatedObs"
       sim_obs.observation_unit = "x"

    You can adjust the **number of particles** for which observations are
    simulated, and the **number of observations** per particle.
    For example, to select 100 particles and simulate 5 observations per
    particle, use the following settings:

    .. code-block:: toml

       [summary.tables]
       sim_obs.component = "pypfilt.summary.SimulatedObs"
       sim_obs.observation_unit = "x"
       sim_obs.particle_count = 100
       sim_obs.observations_per_particle = 5

    This table uses a unique PRNG seed that is derived from the observation
    unit, so that simulated observations for different observation units are
    not correlated.
    It may be desirable to instead use the common PRNG seed (e.g., to preserve
    the existing outputs for scenarios with only a single observation model).
    Set the ``'common_prng_seed'`` setting to ``True`` to enable this:

    .. code-block:: toml

       [summary.tables]
       sim_obs.component = "pypfilt.summary.SimulatedObs"
       sim_obs.observation_unit = "x"
       sim_obs.common_prng_seed = true

    By default, this table includes an ``'fs_time'`` column that records the
    forecasting time.
    This may not always be desirable, and can be avoided by setting the
    ``'include_fs_time'`` setting to ``False``.
    """

    def field_types(self, ctx, obs_list, name):
        # Ensure that a valid observation unit has been specified.
        self.__obs_unit = ctx.get_setting(
            ['summary', 'tables', name, 'observation_unit'])
        if self.__obs_unit is None:
            msg_fmt = 'Summary table {} has no observation unit'
            raise ValueError(msg_fmt.format(name))
        if self.__obs_unit not in ctx.component['obs']:
            msg_fmt = 'Summary table {} has invalid observation unit {}'
            raise ValueError(msg_fmt.format(name, self.__obs_unit))

        # Check whether to include the fs_time column.
        self.__fs_col = ctx.get_setting(
            ['summary', 'tables', name, 'include_fs_time'],
            True)

        # NOTE: each instance of this table uses the same seed for resampling,
        # so that the same particles are selected for each observation unit.
        # However, we use a different seed for simulating the observations for
        # each observation unit, so that they are not correlated.
        prng_seed = ctx.settings['filter'].get('prng_seed')
        self.__rnd = np.random.default_rng(prng_seed)
        # NOTE: use the SHA-1 hash of the observation unit to obtain a PRNG
        # seed that is specific to this observation unit, but is also
        # reproducible.
        # We allow the user to override this setting, because this allows for
        # unchanged outputs when there is only one observation model.
        use_common_seed = ctx.get_setting(
            ['summary', 'tables', name, 'common_prng_seed'],
            False)
        if use_common_seed:
            sim_seed = prng_seed
        else:
            hash_obj = hashlib.sha1(self.__obs_unit.encode('utf-8'))
            hash_val = int.from_bytes(hash_obj.digest(), 'big')
            sim_seed = abs(prng_seed + hash_val)
        self.__sim = np.random.default_rng(sim_seed)

        # Record the number of particles and observations per particle.
        self.__px_count = ctx.get_setting(
            ['summary', 'tables', name, 'particle_count'],
            ctx.particle_count())
        self.__obs_per_px = ctx.get_setting(
            ['summary', 'tables', name, 'observations_per_particle'],
            1)

        # Ensure the observation model does not define a 'fs_time' field, or
        # we cannot record the forecasting time for each observation.
        obs_model = ctx.component['obs'][self.__obs_unit]
        simulated_fields = obs_model.simulated_field_types(ctx)
        for field in simulated_fields:
            if field[0] == 'fs_time':
                msg_fmt = 'Observation model for {} defines "fs_time" field'
                raise ValueError(msg_fmt.format(self.__obs_unit))
        self.__obs_dtype = fields_dtype(ctx, simulated_fields)

        # Add a 'fs_time' column to the simulated observation fields.
        if self.__fs_col:
            fs_time = time_field('fs_time')
            simulated_fields.insert(0, fs_time)

        return simulated_fields

    def load_state(self, ctx, group):
        """Restore the state of each PRNG from the cache."""
        cache.load_rng_states(group, 'prng_states', {
            'resample': self.__rnd,
            'simulate': self.__sim,
        })

    def save_state(self, ctx, group):
        """Save the current state of each PRNG to the cache."""
        cache.save_rng_states(group, 'prng_states', {
            'resample': self.__rnd,
            'simulate': self.__sim,
        })

    def n_rows(self, ctx, forecasting):
        # NOTE: when forecasting, we only want to resample the particles once,
        # at the start of the forecast. This ensures that the simulated
        # observations reflect individual model trajectories. Note that this
        # is not possible during the estimation pass, because any observation
        # can trigger resampling.
        self.__sample_ixs = None
        self.__forecasting = forecasting
        # Need `obs_per_px` rows for each of `n_px` particles.
        n_px = self.__px_count
        obs_per_px = self.__obs_per_px
        n_times = ctx.summary_count()
        return n_times * n_px * obs_per_px

    def add_rows(self, ctx, fs_time, window, insert_fn):
        unit = self.__obs_unit
        obs_model = ctx.component['obs'][unit]

        for snapshot in window:

            # NOTE: resample the particles so that weights are uniform.
            if self.__sample_ixs is None:
                # Select new particle indices.
                (sample_ixs, _weight) = resample.resample_weights(
                    snapshot.weights, self.__rnd,
                    count=self.__px_count)

                # Repeat these indices to simulate multiple observations per
                # particle.
                if self.__obs_per_px > 1:
                    sample_ixs = np.repeat(sample_ixs, self.__obs_per_px)

                # Reuse the same indices for the entire forecasting pass.
                if self.__forecasting:
                    self.__sample_ixs = sample_ixs
            else:
                sample_ixs = self.__sample_ixs

            # Only simulate observations for the selected particles.
            # NOTE: must pass the sample indices to simulated_obs().
            simulated_obs_list = obs_model.simulated_obs(
                ctx, snapshot[sample_ixs], self.__sim)

            for obs in simulated_obs_list:
                obs_row = list(obs_model.obs_into_row(obs, self.__obs_dtype))
                if self.__fs_col:
                    obs_row.insert(0, fs_time)
                insert_fn(tuple(obs_row))


class PredictiveCIs(Table):
    """
    Record fixed-probability central credible intervals for the observations.

    The default intervals are: 0%, 50%, 60%, 70%, 80%, 90%, 95%.
    These can be overridden in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.tables]
       forecasts.component = "pypfilt.summary.PredictiveCIs"
       forecasts.credible_intervals = [ 0, 50, 95 ]
    """

    def __init__(self):
        self.__probs = np.uint8([0, 50, 60, 70, 80, 90, 95])

    def __define_quantiles(self):
        qtl_prs = self.__probs.astype(float) / 100
        qtl_lwr = 0.5 - 0.5 * qtl_prs
        qtl_upr = 0.5 + 0.5 * qtl_prs
        self.__qtls = np.sort(np.unique(np.r_[qtl_lwr, qtl_upr]))

    def field_types(self, ctx, obs_list, name):
        self.__probs = np.uint8(ctx.get_setting(
            ['summary', 'tables', name, 'credible_intervals'],
            self.__probs))
        self.__obs_units = sorted(ctx.component['obs'].keys())
        self.__define_quantiles()
        unit = string_field('unit')
        fs_time = time_field('fs_time')
        time = time_field('time')
        prob = ('prob', np.int8)
        ymin = ('ymin', np.float64)
        ymax = ('ymax', np.float64)
        return [unit, fs_time, time, prob, ymin, ymax]

    def n_rows(self, ctx, forecasting):
        # Need a row for each interval, for each day, for each data source.
        n_obs_models = len(ctx.component['obs'])
        n_times = ctx.summary_count()
        return n_times * len(self.__probs) * n_obs_models

    def add_rows(self, ctx, fs_time, window, insert_fn):
        for unit in self.__obs_units:
            obs_model = ctx.component['obs'][unit]
            for snapshot in window:
                # Quantiles are ordered from smallest to largest.
                qtls = obs_model.quantiles(ctx, snapshot, self.__qtls)
                # Iterate from broadest to narrowest CI.
                for ix, pr in enumerate(self.__probs[::-1]):
                    row = (unit, fs_time, snapshot.time, pr,
                           qtls[ix], qtls[- (ix + 1)])
                    insert_fn(row)


class PartitionPredictiveCIs(Table):
    """
    Record separate fixed-probability central credible intervals for each
    partition in the particle ensemble.

    The default intervals are: 0%, 50%, 60%, 70%, 80%, 90%, 95%.
    These can be overridden in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.tables]
       part_forecasts.component = "pypfilt.summary.PartitionPredictiveCIs"
       part_forecasts.credible_intervals = [ 0, 50, 95 ]
    """

    def __init__(self):
        self.__probs = np.uint8([0, 50, 60, 70, 80, 90, 95])

    def __define_quantiles(self):
        qtl_prs = self.__probs.astype(float) / 100
        qtl_lwr = 0.5 - 0.5 * qtl_prs
        qtl_upr = 0.5 + 0.5 * qtl_prs
        self.__qtls = np.sort(np.unique(np.r_[qtl_lwr, qtl_upr]))

    def field_types(self, ctx, obs_list, name):
        self.__probs = np.uint8(ctx.get_setting(
            ['summary', 'tables', name, 'credible_intervals'],
            self.__probs))
        self.__obs_units = sorted(ctx.component['obs'].keys())
        self.__define_quantiles()
        unit = string_field('unit')
        fs_time = time_field('fs_time')
        time = time_field('time')
        part = ('partition', np.int8)
        prob = ('prob', np.int8)
        ymin = ('ymin', np.float64)
        ymax = ('ymax', np.float64)
        return [unit, fs_time, time, part, prob, ymin, ymax]

    def n_rows(self, ctx, forecasting):
        # Need a row for each interval, for each day, for each data source,
        # for each partition.
        n_obs_models = len(ctx.component['obs'])
        n_parts = len(ctx.settings['filter']['partition'])
        n_times = ctx.summary_count()
        return n_times * len(self.__probs) * n_obs_models * n_parts

    def add_rows(self, ctx, fs_time, window, insert_fn):
        partitions = ctx.settings['filter']['partition']
        for unit in self.__obs_units:
            obs_model = ctx.component['obs'][unit]
            for snapshot in window:
                for pix, partition in enumerate(partitions):
                    ixs = partition['slice']
                    subset = snapshot[ixs]

                    # NOTE: if all particles have zero weight (i.e., this
                    # is a zero-weight reservoir partition) we need to define
                    # non-zero weights.
                    if np.all(subset.weights == 0.0):
                        n = len(subset.weights)
                        subset.weights = np.ones(subset.weights.shape) / n

                    # Quantiles are ordered from smallest to largest.
                    # NOTE: we don't pass `ixs` to quantiles(), because we are
                    # providing exactly those particles whose quantiles we
                    # want to calculate.
                    qtls = obs_model.quantiles(ctx, subset, self.__qtls)
                    # Iterate from broadest to narrowest CI.
                    for ix, pr in enumerate(self.__probs[::-1]):
                        row = (unit, fs_time, snapshot.time, pix + 1,
                               pr, qtls[ix], qtls[- (ix + 1)])
                        insert_fn(row)


class PartitionModelCIs(Table):
    """
    Calculate separate fixed-probability central credible intervals for each
    partition, for all state variables and model parameters.

    .. note:: Credible intervals are only recorded for **scalar** fields.
       Non-scalar fields will be ignored.

    The default intervals are: 0%, 50%, 60%, 70%, 80%, 90%, 95%, 99%, 100%.
    These can be overridden in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.tables]
       part_cints.component = "pypfilt.summary.PartitionModelCIs"
       part_cints.credible_intervals = [ 0, 50, 95 ]
    """

    def __init__(self):
        self.__probs = np.uint8([0, 50, 60, 70, 80, 90, 95, 99, 100])

    def field_types(self, ctx, obs_list, name):
        # Identify the scalar fields.
        self.__field_names = [
            field[0] for field in ctx.component['model'].field_types(ctx)
            if len(field) == 2
        ]
        self.__stat_info = ctx.component['model'].stat_info()
        self.__num_stats = len(self.__field_names) + len(self.__stat_info)
        self.__probs = np.uint8(ctx.get_setting(
            ['summary', 'tables', name, 'credible_intervals'],
            self.__probs))

        fs_time = time_field('fs_time')
        time = time_field('time')
        part = ('partition', np.int8)
        prob = ('prob', np.int8)
        ymin = ('ymin', np.float64)
        ymax = ('ymax', np.float64)
        # State variables/parameters ('model') or statistics ('stat').
        value_type = string_field('type')
        name = string_field('name')
        return [fs_time, time, part, prob, ymin, ymax, value_type, name]

    def n_rows(self, ctx, forecasting):
        # Need a row for each interval, for each day, for each partition, for
        # each parameter, variable and statistic.
        n_parts = len(ctx.settings['filter']['partition'])
        n_times = ctx.summary_count()
        return n_times * len(self.__probs) * n_parts * self.__num_stats

    def add_rows(self, ctx, fs_time, window, insert_fn):
        from . import stats

        partitions = ctx.settings['filter']['partition']
        for snapshot in window:
            for pix, partition in enumerate(partitions):
                ixs = partition['slice']
                subset = snapshot[ixs]

                # NOTE: if all particles have zero weight (i.e., this
                # is a zero-weight reservoir partition) we need to define
                # non-zero weights.
                if np.all(subset.weights == 0.0):
                    n = len(subset.weights)
                    subset.weights = np.ones(subset.weights.shape) / n

                # Identify which state vectors to examine.
                valid = ctx.component['model'].is_valid(
                    subset.state_vec)
                if valid.ndim != 1:
                    raise ValueError('model.is_valid(): expected 1D array')
                if valid.shape != subset.weights.shape:
                    raise ValueError('model.is_valid(): unexpected length')
                # Note that np.where() returns a *tuple*  of arrays (one for
                # each dimension) and we're only scanning a 1D array.
                mask = np.where(valid)[0]
                state_vec = subset.state_vec[mask]

                if np.count_nonzero(mask):
                    ws = subset.weights[mask]
                    for field in self.__field_names:
                        sub_hist = state_vec[field]
                        cred_ints = stats.cred_wt(sub_hist, ws, self.__probs)
                        for cix, pctl in enumerate(self.__probs):
                            row = (fs_time, subset.time, pix + 1, pctl,
                                   cred_ints[pctl][0], cred_ints[pctl][1],
                                   'model', field)
                            insert_fn(row)
                    for (val, stat_fn) in self.__stat_info:
                        stat_vec = stat_fn(state_vec)
                        cred_ints = stats.cred_wt(stat_vec, ws, self.__probs)
                        for cix, pctl in enumerate(self.__probs):
                            row = (fs_time, subset.time, pix + 1, pctl,
                                   cred_ints[pctl][0], cred_ints[pctl][1],
                                   'stat', val)
                            insert_fn(row)
                else:
                    for pctl in self.__probs:
                        for field in self.__field_names:
                            row = (fs_time, subset.time, pix + 1, pctl, 0, 0,
                                   'model', field)
                            insert_fn(row)
                        for (val, _) in self.__stat_info:
                            row = (fs_time, subset.time, pix + 1, pctl, 0, 0,
                                   'stat', val)
                            insert_fn(row)


class BackcastMonitor(Monitor):
    """
    Record the backcast particle matrix at the end of each estimation
    simulation, so that it can be examined in forecasting simulations.

    .. code-block:: toml

       [summary.monitors]
       backcast_monitor.component = "pypfilt.summary.BackcastMonitor"
    """

    backcast = None
    """
    The backcast simulation history (:class:`~pypfilt.state.History`).

    Note that this is **only** valid for tables to inspect during a
    forecasting simulation, and **not** during an estimation simulation.
    """

    window = None
    """
    The backcast summary window (a list of :class:`~pypfilt.state.Snapshot`
    values).

    Note that this is **only** valid for tables to inspect during a
    forecasting simulation, and **not** during an estimation simulation.
    """

    def prepare(self, ctx, obs_list, name):
        self.backcast = None
        self.window = None

    def begin_sim(self, ctx, forecasting):
        self.__forecasting = forecasting

    def end_sim(self, ctx, fs_time, window):
        if self.__forecasting:
            return

        self.backcast = ctx.component['history'].create_backcast(ctx)
        start = self.backcast.times[0]
        until = self.backcast.times[-1]
        self.window = self.backcast.summary_window(ctx, start, until)

    def load_state(self, ctx, group):
        # Attempt to load the cached backcast, if it exists.
        try:
            self.backcast = state.History.load_state(ctx, group)
        except KeyError:
            return
        start = self.backcast.times[0]
        until = self.backcast.times[-1]
        self.window = self.backcast.summary_window(ctx, start, until)

    def save_state(self, ctx, group):
        if self.backcast:
            self.backcast.save_state(ctx, group)


class BackcastPredictiveCIs(Table):
    """
    Record fixed-probability central credible intervals for backcast
    observations.

    This requires a :class:`BackcastMonitor`, which should be specified in
    the scenario settings.

    The default intervals are: 0%, 50%, 60%, 70%, 80%, 90%, 95%.
    These can be overridden in the scenario settings.
    For example:

    .. code-block:: toml

       [summary.monitors]
       backcast_monitor.component = "pypfilt.summary.BackcastMonitor"

       [summary.tables]
       backcasts.component = "pypfilt.summary.BackcastPredictiveCIs"
       backcasts.backcast_monitor = "backcast_monitor"
       backcasts.credible_intervals = [ 0, 50, 95 ]
    """

    def __init__(self):
        self.__probs = np.uint8([0, 50, 60, 70, 80, 90, 95])

    def __define_quantiles(self):
        qtl_prs = self.__probs.astype(float) / 100
        qtl_lwr = 0.5 - 0.5 * qtl_prs
        qtl_upr = 0.5 + 0.5 * qtl_prs
        self.__qtls = np.sort(np.unique(np.r_[qtl_lwr, qtl_upr]))

    def field_types(self, ctx, obs_list, name):
        self.__probs = np.uint8(ctx.get_setting(
            ['summary', 'tables', name, 'credible_intervals'],
            self.__probs))
        self.__monitor_name = ctx.get_setting(
            ['summary', 'tables', name, 'backcast_monitor'])
        self.__monitor = ctx.component['summary_monitor'][self.__monitor_name]
        self.__obs_units = sorted(ctx.component['obs'].keys())
        self.__define_quantiles()
        unit = string_field('unit')
        fs_time = time_field('fs_time')
        time = time_field('time')
        prob = ('prob', np.int8)
        ymin = ('ymin', np.float64)
        ymax = ('ymax', np.float64)
        return [unit, fs_time, time, prob, ymin, ymax]

    def n_rows(self, ctx, forecasting):
        # Need a row for each interval, for each day, for each data source.
        if not forecasting:
            return 0

        n_obs_models = len(ctx.component['obs'])
        n_backcast_times = len(self.__monitor.window)
        return len(self.__probs) * n_backcast_times * n_obs_models

    def add_rows(self, ctx, fs_time, window, insert_fn):
        pass

    def finished(self, ctx, fs_time, window, insert_fn):
        for unit in self.__obs_units:
            obs_model = ctx.component['obs'][unit]
            for snapshot in self.__monitor.window:
                # Quantiles are ordered from smallest to largest.
                qtls = obs_model.quantiles(ctx, snapshot, self.__qtls)
                # Iterate from broadest to narrowest CI.
                for ix, pr in enumerate(self.__probs[::-1]):
                    row = (unit, fs_time, snapshot.time, pr,
                           qtls[ix], qtls[- (ix + 1)])
                    insert_fn(row)


class HDF5(object):
    """
    Save tables of summary statistics to an HDF5 file.

    :param ctx: The simulation context.
    :param obs_list: A list of all observations.
    """

    def __init__(self, ctx):
        obs_list = ctx.all_observations

        # Placeholder for the simulation metadata.
        self.__metadata = None

        # Store the observations.
        self.__all_obs = obs_list

        # Allocate variables to store the details of each summary table.
        self.__tbl_dict = {}
        self.__dtypes = {}
        # When a simulation commences, this will be a dictionary that maps
        # table names to NumPy structured arrays; the value of ``None``
        # indicates that no tables have been allocated.
        self.__df = None

        self.__monitors = {}

        self.__table_group = 'tables'

        # If True when self.__only_fs is True, the current simulation is not a
        # forecasting simulation, and tables should be ignored.
        # Note that monitors are *never* ignored.
        self.__ignore = False

    def initialise(self, ctx):
        """
        Initialise each table and monitor.

        This should be called before running a single set of simulations, and
        is called by each of the top-level pypfilt functions.
        """
        logger = logging.getLogger(__name__)

        # Store simulation metadata.
        if self.__metadata is not None:
            logger.info('Replacing existing summary metadata')
        meta = Metadata()
        self.__metadata = meta.build(ctx)

        # If True, only calculate statistics for forecasting simulations.
        self.__only_fs = ctx.settings['summary']['only_forecasts']

        self.__monitors = {}
        for (name, monitor) in ctx.component['summary_monitor'].items():
            if name in self.__monitors:
                raise ValueError("Monitor '{}' already exists".format(name))
            self.__monitors[name] = monitor
            # NOTE: provide the monitor name here so that the monitor can
            # look for monitor-specific parameters.
            monitor.prepare(ctx, self.__all_obs, name)

        self.__tbl_dict = {}
        for (name, table) in ctx.component['summary_table'].items():
            if name in self.__tbl_dict:
                raise ValueError("Table '{}' already exists".format(name))
            self.__tbl_dict[name] = table
            # NOTE: provide the table name here so that the table can look for
            # table-specific parameters.
            table_fields = table.field_types(ctx, self.__all_obs, name)
            self.__dtypes[name] = fields_dtype(ctx, table_fields)

    def load_state(self, ctx, grp):
        """
        Load the internal state of each monitor and table from a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object from which to load the state.

        :raises ValueError: if a monitor and a table have the same name.
        """
        grp_names = set()

        # Load the monitor states.
        for (name, mon) in self.__monitors.items():
            if name in grp_names:
                msg = "Multiple {} monitors/tables".format(name)
                raise ValueError(msg)
            else:
                grp_names.add(name)
            mon_grp = grp.require_group(name)
            mon.load_state(ctx, mon_grp)

        # Load the table states.
        for (name, tbl) in self.__tbl_dict.items():
            if name in grp_names:
                msg = "Multiple {} monitors/tables".format(name)
                raise ValueError(msg)
            else:
                grp_names.add(name)
            tbl_grp = grp.require_group(name)
            tbl.load_state(ctx, tbl_grp)

    def save_state(self, ctx, grp):
        """
        Save the internal state of each monitor and table to a cache file.

        :param ctx: The simulation context.
        :param grp: The h5py Group object in which to save the state.

        :raises ValueError: if a monitor and a table have the same name.
        """
        grp_names = set()

        # Save the monitor states.
        for (name, mon) in self.__monitors.items():
            if name in grp_names:
                msg = "Multiple {} monitors/tables".format(name)
                raise ValueError(msg)
            else:
                grp_names.add(name)
            mon_grp = grp.require_group(name)
            mon.save_state(ctx, mon_grp)

        # Save the table states.
        for (name, tbl) in self.__tbl_dict.items():
            if name in grp_names:
                msg = "Multiple {} monitors/tables".format(name)
                raise ValueError(msg)
            else:
                grp_names.add(name)
            tbl_grp = grp.require_group(name)
            tbl.save_state(ctx, tbl_grp)

    def allocate(self, ctx, forecasting=False):
        """
        Allocate space for the simulation statistics.

        This is called by ``pypfilt.pfilter.run`` before running a simulation.
        When multiple simulations are run, such as a series of forecasts, this
        will be called before each simulation.

        :param ctx: The simulation context.
        :param forecasting: Whether this is an estimation pass (default) or a
            forecasting pass.
        """
        if self.__df is not None:
            raise ValueError("Tables have already been allocated")

        if self.__only_fs and not forecasting:
            # Flag this simulation as being ignored.
            self.__ignore = True
        else:
            self.__ignore = False

        logger = logging.getLogger(__name__)

        start_time = ctx.settings['time']['sim_start']
        end_time = ctx.settings['time']['sim_until']
        self.__start_time = start_time
        self.__end_time = end_time
        if forecasting:
            # Forecasting from the start of the simulation period.
            self.__fs_time = self.__start_time
        else:
            # Not forecasting, so all observations are included.
            # NOTE: set this to the end of the *true* simulation period.
            # This avoids a problem where, when using a cache file, the
            # estimation run will end at the final forecasting data and both
            # the estimation run and this final forecast will be identified by
            # the same "forecasting" time.
            self.__fs_time = ctx.settings['time']['until']

        # Notify each monitor, regardless of whether tables are ignored.
        for mon in self.__monitors.values():
            mon.begin_sim(ctx, forecasting)

        n_times = len(list(ctx.summary_times()))
        if self.__ignore:
            logger.debug("Summary.allocate({}, {}): {} points".format(
                start_time, end_time, n_times))
            return

        # Determine the number of rows to allocate for each table.
        # Note: the items() method returns a list in Python 2 and a view
        # object in Python 3; since the number of tables will always be small
        # (on the order of 10) the overhead of using items() in Python 2 and
        # not iteritems() --- which returns an interator --- is negligible.
        n_rows = {n: t.n_rows(ctx, forecasting)
                  for (n, t) in self.__tbl_dict.items()}
        # Allocate tables that require at least one row.
        self.__df = {tbl: np.empty(n_rows[tbl], dtype=self.__dtypes[tbl])
                     for tbl in n_rows if n_rows[tbl] > 0}
        # Initialise a row counter for each allocated table.
        self.__ix = {tbl: 0 for tbl in self.__df}
        # Create a row insertion function for each allocated table.
        self.__insert = {tbl: self.__insert_row_fn(tbl) for tbl in self.__df}

        logger.debug("Summary.allocate({}, {}): {} points".format(
            start_time, end_time, n_times))

    def __insert_row_fn(self, tbl):
        def insert(fields, n=1):
            row_ix = self.__ix[tbl]
            self.__df[tbl][row_ix:row_ix + n] = fields
            self.__ix[tbl] += n
        return insert

    def summarise(self, ctx, window):
        """
        Calculate statistics for some portion of the simulation period.

        :param ctx: The simulation context.
        :param window: A list of :class:`~pypfilt.state.Snapshot` instances
            that capture the particle states at each summary time in the
            simulation window.
        """
        logger = logging.getLogger(__name__)

        if self.__df is None and not self.__ignore:
            raise ValueError("Tables have not been allocated")

        num_times = len(window)
        if num_times == 0:
            logger.debug("Summary.summarise: no times in window")
            return

        start_time = window[0].time
        end_time = window[-1].time

        if self.__start_time > start_time or self.__end_time < end_time:
            raise ValueError("Summary.summarise() called for invalid period")

        logger.debug("Summary.summarise({}, {}): {} times".format(
            start_time, end_time, num_times))

        fs_time = self.__fs_time

        for mon in self.__monitors.values():
            mon.monitor(ctx, fs_time, window)

        tables = self.__df if self.__df is not None else []

        for tbl in tables:
            insert_fn = self.__insert[tbl]
            self.__tbl_dict[tbl].add_rows(ctx, fs_time, window, insert_fn)

        if end_time == self.__end_time:
            for mon in self.__monitors.values():
                mon.end_sim(ctx, fs_time, window)

            for tbl in tables:
                insert_fn = self.__insert[tbl]
                self.__tbl_dict[tbl].finished(ctx, fs_time, window, insert_fn)

    def get_stats(self):
        """Return the calculated statistics for a single simulation."""
        if self.__df is None:
            if self.__ignore:
                return {}
            else:
                raise ValueError("Tables have not been created")

        logger = logging.getLogger(__name__)
        logger.debug("Summary.get()")

        # Check all table rows are filled (and no further).
        for tbl in self.__df:
            alloc = self.__df[tbl].shape[0]
            used = self.__ix[tbl]
            if alloc != used:
                msg = "Table '{}' allocated {} rows but filled {}"
                raise ValueError(msg.format(tbl, alloc, used))

        # Return the summary tables and remove them from this class instance.
        stats = self.__df
        self.__df = None
        return stats

    def save_forecasts(self, ctx, results, filename):
        """
        Save forecast summaries to disk in the HDF5 binary data format.

        This function creates the following datasets that summarise the
        estimation and forecasting outputs:

        - ``'tables/TABLE'`` for each table.

        The provided metadata will be recorded under ``'meta/'``.

        If dataset creation timestamps are enabled, two simulations that
        produce identical outputs will not result in identical files.
        Timestamps will be disabled where possible (requires h5py >= 2.2):

        - ``'hdf5_track_times'``: Presence of creation timestamps.

        :param ctx: The simulation context.
        :param results: The simulation results.
        :param filename: The filename to which the data will be written.
        """
        fs_times = results.forecast_times()
        table_names = set()
        for fs_time in fs_times:
            for table_name in results.forecasts[fs_time].tables:
                table_names.add(table_name)
        if results.estimation is not None:
            for table_name in results.estimation.tables:
                table_names.add(table_name)

        # Construct aggregate data tables.
        # Note that some tables may not exist for every simulation.
        tbl_dict = {}
        for name in table_names:
            sub_tables = [
                results.forecasts[fs_time].tables[name]
                for fs_time in fs_times
                if name in results.forecasts[fs_time].tables]
            if results.estimation is not None:
                if name in results.estimation.tables:
                    sub_tables.append(results.estimation.tables[name])
            tbl_dict[name] = np.concatenate(sub_tables)

        # Do not record dataset creation timestamps.
        kwargs = {'track_times': False}

        time_scale = ctx.component['time']

        def save_data(g, name, value):
            if isinstance(value, dict):
                sub_g = g.create_group(name)
                for sub_key, sub_value in sorted(value.items()):
                    save_data(sub_g, sub_key, sub_value)
            else:
                # NOTE: ensure that all time values are in a format that can
                # be stored in a HDF5 dataset.
                try:
                    if isinstance(value, np.ndarray):
                        save_dataset(time_scale, g, name, value, **kwargs)
                    else:
                        g.create_dataset(name, data=value, **kwargs)
                except TypeError:
                    msg = 'Error saving dataset "{}" with value {} and type {}'
                    raise ValueError(msg.format(name, value,
                                                type(value).__name__))

        with h5py.File(filename, 'w') as f:
            # Save the associated metadata, if any.
            if self.__metadata:
                meta_grp = f.create_group('meta')
                for k, v in sorted(self.__metadata.items()):
                    save_data(meta_grp, k, v)

            # Compress and checksum the data tables.
            kwargs['compression'] = 'gzip'
            kwargs['shuffle'] = True
            kwargs['fletcher32'] = True

            # Save the data tables.
            tbl_grp = f.create_group(self.__table_group)
            for tbl in tbl_dict:
                save_data(tbl_grp, tbl, tbl_dict[tbl])

            # Save the adaptive fit tables, if they exist.
            adaptive_group = 'adaptive_fit'
            if results.adaptive_fits:
                adapt_group = f.create_group(adaptive_group)
                # Concatenate tables from each pass, and add an
                # exponent field.
                fits = results.adaptive_fits
                first_fit = next(iter(fits.values()))
                for n in table_names:
                    # Concatenate the tables from each pass, adding a new
                    # field that records the adaptive fit exponent.
                    table = np.concatenate([
                        nprec.append_fields(
                            fit.tables[n],
                            'adaptive_fit_exponent',
                            exponent * np.ones(
                                fit.tables[n].shape),
                            usemask=False)
                        for (exponent, fit) in fits.items()])
                    # Ensure that the dtype metadata (if any) is preserved.
                    metadata = first_fit.tables[n].dtype.metadata
                    if metadata is not None:
                        table = table.view(dtype=np.dtype(
                            table.dtype, metadata=dict(metadata)))
                    save_data(adapt_group, n, table)

            # Save the observation tables in ctx.data['obs'].
            obs_group = f.create_group('observations')
            for (key, value) in ctx.data['obs'].items():
                save_data(obs_group, key, value)

            # Save the prior sample tables in ctx.data['prior'].
            prior_group = f.create_group('prior_samples')
            for (key, value) in ctx.data['prior'].items():
                save_data(prior_group, key, value)

            # Save the history matrix state if instructed to do so.
            if ctx.get_setting(['summary', 'save_history'], False):
                history = ctx.component['history']
                history_group = f.create_group('history')
                save_data(history_group, 'matrix', history.matrix)
                save_data(history_group, 'times', history.times)

            # Save the backcast if instructed to do so.
            if ctx.get_setting(['summary', 'save_backcast'], False):
                history = ctx.component['history']
                backcast = history.create_backcast(ctx)
                backcast_group = f.create_group('backcast')
                save_data(backcast_group, 'matrix', backcast.matrix)
                save_data(backcast_group, 'times', backcast.times)


class Metadata(object):
    """
    Document the simulation settings and system environment for a set of
    simulations.
    """

    def build(self, ctx):
        """
        Construct a metadata dictionary that documents the simulation
        parameters and system environment. Note that this should be generated
        at the **start** of the simulation, and that the git metadata will
        only be valid if the working directory is located within a git
        repository.

        :param ctx: The simulation context.

        By default, the versions of ``pypfilt``, ``h5py``, ``numpy`` and
        ``scipy`` are recorded.
        """

        # Import modules for extracting system-specific details.
        import platform
        # Import modules for extracting package versions.
        import scipy
        import lhs
        # Import modules for parsing scenario configurations.
        import tomli
        import tomli_w

        # Record the command line used to launch this simulation.
        # Note that sys.argv is a list of native strings.
        cmdline = " ".join(sys.argv)

        meta = {
            'package': {
                'python': platform.python_version(),
                'h5py': self.pkg_version(h5py),
                'lhs': self.pkg_version(lhs),
                'pypfilt': self.pkg_version(version),
                'numpy': self.pkg_version(np),
                'scipy': self.pkg_version(scipy),
                'tomli': self.pkg_version(tomli),
                'tomli-w': self.pkg_version(tomli_w),
            },
            'sim': {
                'cmdline': cmdline,
            },
            'settings': self.encode_settings(ctx.settings, self.encode),
        }

        git_data = self.git_data()
        if git_data:
            meta['git'] = git_data

        # Record the versions of user-specified packages (if any).
        pkgs = ctx.get_setting(
            ['summary', 'metadata', 'packages'], [])
        for package_name in pkgs:
            mod_obj = importlib.import_module(package_name)
            meta['package'][package_name] = self.pkg_version(mod_obj)

        return meta

    def encode_settings(self, values, encode_fn):
        """
        Recursively encode settings in a dictionary.

        :param values: The original dictionary.
        :param encode_fn: A function that encodes individual values (see
            :func:`.encode_value`).
        """
        retval = {}
        for k, v in values.items():
            if isinstance(v, dict):
                # Recursively encode this dictionary.
                retval[k] = self.encode_settings(v, encode_fn)
            else:
                retval[k] = encode_fn(v)
        return retval

    def encode(self, value):
        """
        Encode values in a form suitable for serialisation in HDF5 files.

        * Integer values are converted to ``numpy.int32`` values.
        * Floating-point values and arrays retain their data type.
        * All other (i.e., non-numerical) values are converted to UTF-8
          strings.
        """
        if isinstance(value, (int, np.int64)):
            # Avoid storing 64-bit integers since R doesn't support them.
            return np.int32(value)
        elif isinstance(value, (float, np.ndarray)):
            # Ensure that numerical values retain their data type.
            return value
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            # Save numeric lists and tuples as NumPy arrays.
            all_numbers = all(
                isinstance(v, (int, float, np.int_, np.float_))
                for v in value)
            if all_numbers:
                return np.array(value)
            else:
                return str(value)
        else:
            # Convert non-numerical values to UTF-8 strings.
            return str(value)

    def pkg_version(self, module):
        """Attempt to obtain the version of a Python module."""
        try:
            return str(module.__version__)
        except AttributeError:
            try:
                # Older versions of h5py store the version number here.
                return str(module.version.version)
            except AttributeError:
                return "unknown"

    def git_data(self):
        """
        Record the status of the git repository within which the working
        directory is located (if such a repository exists).
        """
        # Determine the encoding for the default locale.
        default_encoding = locale.getdefaultlocale()[1]
        logger = logging.getLogger(__name__)
        enc_msg = "Extracting git metadata using locale encoding '{}'"
        logger.debug(enc_msg.format(default_encoding))

        # Return no metadata if git is not installed, or if the working
        # directory is not located within a git repository.
        try:
            git_head = self.run_cmd(['git', 'rev-parse', 'HEAD'])
        except FileNotFoundError:
            logger.info('Could not run git; do you have git installed?')
            return {}
        if not git_head:
            logger.info('No HEAD commit; presumably not in a git repository?')
            return {}

        git_branch = self.run_cmd(['git', 'symbolic-ref', '--short', 'HEAD'])
        git_mod_files = self.run_cmd(['git', 'ls-files', '--modified'],
                                     all_lines=True, err_val=[])
        git_mod_files.sort()
        git_mod = len(git_mod_files) > 0
        return {
            'HEAD': git_head,
            'branch': git_branch,
            'modified': git_mod,
            'modified_files': [f.encode("utf-8") for f in git_mod_files],
        }

    def run_cmd(self, args, all_lines=False, err_val=''):
        """
        Run a command and return the (Unicode) output. By default, only the
        first line is returned; set ``all_lines=True`` to receive all of the
        output as a list of Unicode strings. If the command returns a non-zero
        exit status, return ``err_val`` instead.
        """
        try:
            # Return the output as a single byte string.
            lines = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            return err_val
        # Decode and break into lines according to Unicode boundaries.
        # For details see:
        # * https://docs.python.org/2/library/stdtypes.html#unicode.splitlines
        # * https://docs.python.org/3/library/stdtypes.html#str.splitlines
        default_encoding = locale.getdefaultlocale()[1]
        lines = lines.decode(default_encoding).splitlines()
        if all_lines:
            return lines
        else:
            return lines[0]
