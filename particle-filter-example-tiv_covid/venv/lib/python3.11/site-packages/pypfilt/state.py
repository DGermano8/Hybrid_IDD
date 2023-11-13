"""Manage particle state vectors and their history."""

import logging
import numpy as np
import numpy.lib.recfunctions


def is_history_matrix(ctx, arr):
    """
    Check whether the provided array includes all columns (including, e.g.,
    the particle weights and parent indices).

    :param ctx: The simulation context.
    :param arr: The array to check.

    :returns: ``True`` if the check is successful, otherwise ``False``.
    """
    expected_dtype = np.dtype(history_matrix_dtype(ctx))
    return arr.dtype == expected_dtype


def is_state_vec_matrix(ctx, arr):
    """
    Check whether the provided array includes only the particle state vector
    columns.

    :param ctx: The simulation context.
    :param arr: The array to check.

    :returns: ``True`` if the check is successful, otherwise ``False``.
    """
    expected_dtype = np.dtype(state_vec_dtype(ctx))
    return arr.dtype == expected_dtype


def require_history_matrix(ctx, arr):
    """
    Check whether the provided array includes all columns (including, e.g.,
    the particle weights and parent indices).

    :param ctx: The simulation context.
    :param arr: The array to check.

    :raises ValueError: if the check fails.
    """
    if not is_history_matrix(ctx, arr):
        raise ValueError('unexpected dtype: {}'.format(arr.dtype))


def require_state_vec_matrix(ctx, arr):
    """
    Check whether the provided array includes only the particle state vector
    columns.

    :param ctx: The simulation context.
    :param arr: The array to check.

    :raises ValueError: if the check fails.
    """
    if not is_state_vec_matrix(ctx, arr):
        raise ValueError('unexpected dtype: {}'.format(arr.dtype))


def state_vec_dtype(ctx):
    """
    Returns the dtype of the state vector matrix.

    Note that this is returned as a list, not  a ``numpy.dtype`` value.
    """
    sv_dtype = ctx.component['model'].field_types(ctx)
    # NOTE: h5py doesn't allow zero-sized dimensions, so state_vec must
    # contain *something*; see https://github.com/h5py/h5py/issues/944.
    # Here we use a Boolean (single byte); by default it will be `False`.
    if len(sv_dtype) == 0:
        return [('state_vec', np.bool_)]
    return [('state_vec', sv_dtype)]


def history_matrix_dtype(ctx):
    """
    Returns the dtype of the particle history matrix.

    Note that this is returned as a list, not  a ``numpy.dtype`` value.

    The top-level fields are:

    * weight: the particle weights (float);
    * prev_ix: the parent indices (int);
    * state_vec: the particle state vectors (float or structured dtype); and
    * lookup: sample indices for lookup tables (if required).
    """
    # We always need to record the particle weight and parent index.
    base_dtype = [('weight', np.float_), ('prev_ix', np.int_),
                  ('resampled', np.bool_)]

    # Determine the structure of the model state vector.
    svec_dtype = state_vec_dtype(ctx)

    # Determine whether we need additional lookup columns.
    # If no lookup columns are required we need to avoid adding the 'lookup'
    # field, because it will have zero dimensions and we will be unable to
    # save the history matrix in HDF5 files (e.g., cache files).
    lookup_cols = ctx.data.get('lookup_ixs', {})
    if lookup_cols:
        lookup_dtype = [('lookup', [(name, np.int_) for name in lookup_cols])]
        hist_dtype = base_dtype + svec_dtype + lookup_dtype
    else:
        hist_dtype = base_dtype + svec_dtype

    return hist_dtype


def history_matrix(ctx):
    """
    Allocate a particle history matrix of sufficient size to store an entire
    particle filter simulation.

    :param ctx: The simulation context.

    :returns: A particle history matrix.
    :rtype: numpy.ndarray
    """
    # Determine the number of particles and their initial weights.
    px_count = ctx.settings['filter']['particles']
    if px_count < 1:
        raise ValueError("Too few particles: {}".format(px_count))
    logger = logging.getLogger(__name__)
    logger.debug("Size = {}".format(px_count))

    # Determine the number of time-steps for which to allocate space.
    # NOTE: if we want history_window to define the **allocated size** then we
    # need to enforce history_window >= 3 and then define the window size as
    # ``(history_window - 1) / 2``.
    window_size = ctx.settings['filter']['history_window']
    if window_size > 0:
        steps = ctx.settings['time']['steps_per_unit']
        num_steps = 2 * window_size * steps + 1
    else:
        # NOTE: we cannot use ``time_scale.step_count()`` to determine how
        # large to make the history matrix so that it can store the entire
        # simulation period.
        # This is because the current simulation period may be, e.g., a
        # truncated estimation pass that ends at the forecasting time.
        # However, we can inspect the original simulation parameters and count
        # the number of time-steps required to reach the end of the entire
        # simulation period.
        until = ctx.settings['time']['until']
        num_time_steps = ctx.component['time'].step_of(until)
        num_steps = num_time_steps + 1

    # Allocate the particle history matrix and record the initial states.
    hist_size = (num_steps, px_count)
    hist_dtype = history_matrix_dtype(ctx)
    hist = np.zeros(hist_size, dtype=hist_dtype)
    logger.debug("Hist.nbytes = {}".format(hist.nbytes))

    # Initialise all lookup columns.
    lookup_cols = ctx.data.get('lookup_ixs', {})
    for lookup_name, values in lookup_cols.items():
        hist[0, :]['lookup'][lookup_name] = values

    # Initialise the particle weights, parent indices, and state vectors.
    for partition in ctx.settings['filter']['partition']:
        ixs = partition['slice']
        init_weight = partition['weight'] / partition['particles']
        hist['weight'][0, ixs] = init_weight
    hist['prev_ix'][0] = np.arange(ctx.settings['filter']['particles'])
    ctx.component['model'].init(ctx, hist['state_vec'][0])

    # Return the allocated (and initialised) particle history matrix.
    return hist


def earlier_states(hist, ix, steps):
    """
    Return the particle states at a previous time-step, ordered with respect
    to their current arrangement.

    :param hist: The particle history matrix.
    :param ix: The current time-step index.
    :param steps: The number of steps back in time.
    """
    logger = logging.getLogger(__name__)

    # Return the current particles when looking back zero time-steps.
    if steps == 0:
        logger.debug('Looking back zero steps')
        return hist[ix - steps]

    # Don't go too far back (negative indices jump into the future).
    if steps > ix:
        msg_fmt = 'Cannot look back {} time-steps, will look back {}'
        logger.debug(msg_fmt.format(steps, ix))
    steps = min(steps, ix)

    # Start with the parent indices for the current particles, which allow us
    # to look back one time-step.
    parent_ixs = np.copy(hist['prev_ix'][ix])

    # Continue looking back one time-step, and only update the parent indices
    # if the particles were resampled.
    for i in range(1, steps):
        step_ix = ix - i
        if hist['resampled'][step_ix, 0]:
            parent_ixs = hist['prev_ix'][step_ix, parent_ixs]

    return hist[ix - steps, parent_ixs]


def repack(svec, astype=float):
    """
    Return a copy of the array ``svec`` where the fields are contiguous and
    viewed as a regular Numpy array of ``astype``.

    :raises ValueError: if ``svec`` contains any fields that are incompatible
        with ``astype``.

    :Examples:

    >>> import numpy as np
    >>> from pypfilt.state import repack
    >>> xs = np.array([(1.2, (2.2, 3.2)), (4.2, (5.2, 6.2))],
    ...               dtype=[('x', float), ('y', float, 2)])
    >>> ys = repack(xs)
    >>> assert np.array_equal(ys, np.array([[1.2, 2.2, 3.2],
    ...                                     [4.2, 5.2, 6.2]]))
    """

    def is_compat(dt):
        if dt.subdtype is None:
            return np.issubdtype(dt, astype)
        else:
            return np.issubdtype(dt.subdtype[0], astype)

    field_dtypes = {name: info[0] for name, info in svec.dtype.fields.items()}
    incompat = [name for name, dt in field_dtypes.items()
                if not is_compat(dt)]
    if incompat:
        msg = 'Fields {} are not compatible with type {}'
        raise ValueError(msg.format(', '.join(incompat), astype))

    out = numpy.lib.recfunctions.structured_to_unstructured(svec)
    return out.view(astype)


class Snapshot:
    """
    Captures the particle states at a single moment in time.

    The following instance variables are defined:

    + **time** – The simulation time (``time``, below).
    + **hist_ix** – The index of this snapshot (``hist_ix``, below).
    + **weights** – The array of particle weights.
    + **vec** – The history matrix slice for this time.
    + **state_vec** – The particle state vectors at this time.

    :param time: The simulation time.
    :param steps_per_unit: The number of time-steps per unit time.
    :param matrix: The history matrix.
    :param hist_ix: The index of this snapshot in the history matrix.
    """
    def __init__(self, time, steps_per_unit, matrix, hist_ix):
        self.time = time
        self.hist_ix = hist_ix
        self.weights = matrix['weight'][hist_ix]
        self.vec = matrix[hist_ix]
        self.state_vec = matrix[hist_ix]['state_vec']
        self.__matrix = matrix
        self.__steps_per_unit = steps_per_unit
        self.__slice = None

    def __len__(self):
        """
        The snapshot length is the number of particles in the snapshot.
        """
        return self.vec.shape[0]

    def __getitem__(self, key):
        """
        Return a new snapshot the captures a subset of the ensemble.

        :param key: The indices into the particle state vectors.
            This must be a slice, an integer, or a sequence of integers.
        :type key: Union[slice, int, np.ndarray]
        """
        if self.sliced():
            raise ValueError('Cannot slice a sliced Snapshot')

        if isinstance(key, slice):
            pass
        elif isinstance(key, int):
            # Convert integer keys into 1D arrays.
            key = np.array([key])
        elif not isinstance(key, np.ndarray):
            # Try converting all other types into NumPy arrays.
            try:
                key = np.array(key)
            except TypeError:
                raise KeyError(key)

        # Ensure that if key is an array, it is one-dimensional.
        if isinstance(key, np.ndarray) and key.ndim != 1:
            raise KeyError(key)

        if self.__slice is None:
            subset = Snapshot(self.time, self.__steps_per_unit,
                              self.__matrix, self.hist_ix)
            subset.__slice = key
            subset.vec = subset.vec[key]
            subset.state_vec = subset.state_vec[key]
            subset.weights = subset.weights[key]
            return subset

    def sliced(self):
        """
        Return ``True`` if the snapshot captures a slice of the ensemble.
        """
        return self.__slice is not None

    def back_n_steps(self, n):
        """
        Return the history matrix slice ``n`` time-steps before this snapshot.

        .. note:: This returns an array, **not** a :class:`Snapshot`.

        :param n: The number of time-steps to step back.
        """
        hist = earlier_states(self.__matrix, self.hist_ix, n)
        if self.__slice is None:
            return hist
        else:
            return hist[self.__slice]

    def back_n_steps_state_vec(self, n):
        """
        Return the particle state vectors ``n`` time-steps before this
        snapshot.

        .. note:: This returns the state vectors as an array, **not** as a
           :class:`Snapshot`.

        :param n: The number of time-steps to step back.
        """
        return self.back_n_steps(n)['state_vec']

    def back_n_units(self, n):
        """
        Return the history matrix slice ``n`` time units before this snapshot.

        .. note:: This returns an array, **not** a :class:`Snapshot`.

        :param n: The number of time units to step back.
        """
        steps = n * self.__steps_per_unit
        return self.back_n_steps(steps)

    def back_n_units_state_vec(self, n):
        """
        Return the particle state vectors ``n`` time units before this
        snapshot.

        .. note:: This returns the state vectors as an array, **not** as a
           :class:`Snapshot`.

        :param n: The number of time units to step back.
        """
        return self.back_n_units(n)['state_vec']


class History:
    """
    A simulation history stores the current and past states of each particle,
    as well as each particle's weight and parent index.

    .. note:: The ``matrix`` and ``offset`` arguments should only be provided
       when restoring a cached history matrix.

    :param ctx: The simulation context.
    :param matrix: The (optional) history matrix; a new matrix will be created
        if this is ``None`` (default behaviour).
    :param offset: The (optional) offset that maps time step numbers to slices
        of the history matrix.
    :param times: The (optional) list of times associated with each slice of
        the history matrix.
    """

    def __init__(self, ctx, matrix=None, offset=0, times=None):
        if matrix is None:
            matrix = history_matrix(ctx)

        # Ensure that the history matrix has the expected dimensions and type.
        num_particles = ctx.settings['filter']['particles']
        if matrix.shape[1] != num_particles:
            msg_fmt = 'Invalid history matrix size {} for {} particles'
            raise ValueError(msg_fmt.format(matrix.shape, num_particles))
        require_history_matrix(ctx, matrix)

        self.matrix = matrix
        self.offset = offset
        self.index = None
        if times is None:
            self.times = [ctx.settings['time']['sim_start']]
        else:
            self.times = times

    def create_backcast(self, ctx):
        """
        Return a new simulation history that represents the backcast from the
        current particle states, extending as far back as the oldest recorded
        particle states.

        :param ctx: The simulation context.

        .. note:: All particle states in the backcast will be assigned the
           current particle weights.
        """
        # Copy the current history matrix.
        backcast = np.copy(self.matrix[:self.index + 1])
        # Loop over the parent states, and update their ordering.
        for parent_ix in reversed(range(self.index)):
            child_ix = parent_ix + 1
            ixs = backcast['prev_ix'][child_ix]
            backcast[parent_ix] = backcast[parent_ix, ixs]

        # Apply the current particle weights to the entire backcast.
        backcast['weight'][:self.index] = backcast['weight'][self.index]
        # Remove any resampling indicators, they are not relevant.
        backcast['resampled'] = False
        # Reset the parent indices.
        backcast['prev_ix'] = np.arange(backcast.shape[1])

        # Ensure we set the backcast time-step index appropriately.
        history = History(ctx, matrix=backcast, times=list(self.times),
                          offset=self.offset)
        history.index = self.index
        return history

    @staticmethod
    def load_state(ctx, group):
        """
        Load the history matrix state from a cache file and return a new
        ``History`` object.

        :param ctx: The simulation context.
        :param group: The h5py Group object from which to load the history
            matrix state.
        """
        matrix = group['matrix'][()]
        offset = group['offset'][()]
        times = group['times'][()]
        # NOTE: must convert time values to native format.
        time_scale = ctx.component['time']
        times = [time_scale.from_dtype(t) for t in times]
        # NOTE: ensure the starting index is defined so that we can, e.g.,
        # create snapshots of the initial state.
        hist = History(ctx, matrix, offset, times)
        hist.index = hist.offset
        return hist

    def save_state(self, ctx, group):
        """
        Save the history matrix state to a cache file.

        :param ctx: The simulation context.
        :param group: The h5py Group object in which to save the history
            matrix state.
        """
        # NOTE: must convert time values to stored format.
        time_scale = ctx.component['time']
        times = np.array([time_scale.to_dtype(t)
                          for t in self.times])
        # NOTE: we must save the current index as the offset!
        data_sets = {
            'matrix': self.matrix,
            'offset': self.index,
            'times': times,
        }
        # Remove any existing data sets so that we can replace them.
        for (name, data) in data_sets.items():
            if name in group:
                del group[name]
            try:
                dtype = getattr(data, 'dtype', None)
                group.create_dataset(name, data=data, dtype=dtype)
            except Exception:
                logger = logging.getLogger(__name__)
                msg_fmt = 'Could not create dataset {}/{}'
                logger.error(msg_fmt.format(group.name, name))
                msg_fmt = 'Data has dtype: {}'
                logger.error(msg_fmt.format(dtype))
                raise

    def set_time_step(self, step_num, time):
        """
        Update ``history.index`` to point to the specified time step.
        """
        self.index = step_num + self.offset

        # Record the time associated with this slice of the matrix.
        if self.index < len(self.times):
            self.times[self.index] = time
        elif self.index == len(self.times):
            self.times.append(time)
        else:
            msg_fmt = 'Time step {} at {} has unexpected index {}'
            raise ValueError(msg_fmt.format(step_num, time, self.index))

        return self.index

    def reached_window_end(self):
        """
        Return ``True`` if the current time step lies outside the current
        history window, in which case the window must be shifted before the
        time step can be simulated.
        """
        return self.index >= self.matrix.shape[0]

    def shift_window_back(self, shift):
        """
        Shift the history window by ``shift`` steps.
        """
        self.offset -= shift
        self.index -= shift
        self.matrix[:-shift, :] = self.matrix[shift:, :]
        self.matrix['state_vec'][self.index] = 0
        self.times = self.times[shift:]
        if self.matrix.dtype.names is not None:
            if 'lookup' in self.matrix.dtype.names:
                self.matrix['lookup'][self.index] = 0

    def set_resampled(self, was_resampled):
        """
        Record whether the particles were resampled at the current time step.
        """
        self.matrix['resampled'][self.index] = was_resampled

    def summary_window(self, ctx, start, end):
        """
        Return a list of :class:`Snapshot` values for each time unit in the
        summary window, from ``start`` to ``end`` (inclusive).

        :param ctx: The simulation context.
        :param start: The starting time for the summary window (inclusive).
        :param end: The ending time for the summary window (inclusive).
        """
        steps_per_unit = ctx.settings['time']['steps_per_unit']
        window_times = [
            (time_val, step_num + self.offset)
            for (step_num, time_val) in ctx.summary_times()
            if start <= time_val <= end]

        return [Snapshot(time, steps_per_unit, self.matrix, hist_ix)
                for (time, hist_ix) in window_times]

    def snapshot(self, ctx, expect_time=None):
        """
        Return a :class:`Snapshot` of the current particle states.

        :param ctx: The simulation context.
        :param expected_time: The expected value for the current time
            (optional).

        :raises ValueError: if the snapshot time differs from ``expect_time``.
        """
        if self.index is None:
            msg = 'Cannot create snapshot without a time-step index'
            raise ValueError(msg)
        time = self.times[-1]
        if expect_time is not None and time != expect_time:
            msg_fmt = 'Snapshot time is {}, but expected {}'
            raise ValueError(msg_fmt.format(time, expect_time))
        steps_per_unit = ctx.settings['time']['steps_per_unit']
        return Snapshot(time, steps_per_unit, self.matrix, self.index)

    def previous_snapshot(self, ctx, expect_time=None):
        """
        Return a :class:`Snapshot` of the previous particle states.

        :param ctx: The simulation context.
        :param expected_time: The expected value for the previous time
            (optional).

        :raises ValueError: if the snapshot time differs from ``expect_time``.
        """
        if self.index is None:
            msg = 'Cannot create snapshot without a time-step index'
            raise ValueError(msg)
        if len(self.times) >= 2:
            time = self.times[-2]
        else:
            time = ctx.settings['time']['start']
        if expect_time is not None and time != expect_time:
            msg_fmt = 'Snapshot time is {}, but expected {}'
            raise ValueError(msg_fmt.format(time, expect_time))
        steps_per_unit = ctx.settings['time']['steps_per_unit']
        return Snapshot(time, steps_per_unit, self.matrix, self.index - 1)
