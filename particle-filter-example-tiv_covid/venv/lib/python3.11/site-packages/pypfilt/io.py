"""Read data from external sources."""

import datetime
import h5py
import logging
import numpy as np
import os
import os.path


def read_table(path, columns, comment='#', encoding='utf-8'):
    """
    Read data from a space-delimited text file with column headers defined in
    the first non-comment line.

    .. warning::

       This **does not handle string columns**.
       To load tabular data that includes string values, you should use
       `numpy.genfromtxt`_ and then change the array data type:

       .. code-block:: python

          import numpy as np
          import pypfilt
          from pypfilt.io import time_field, string_field, fields_dtype

          # Load data from a text file.
          filename = 'input.ssv'
          table = np.genfromtxt(filename, dtype=None)

          # Define the table fields.
          fields = [time_field('time'), string_field('location')]
          time_scale = pypfilt.Datetime()
          dtype = fields_dtype(time_scale, fields)

          # Change the array data type.
          table = table.asdtype(dtype)

    :param path: The path to the data file.
    :param columns: The columns to read from the data file, represented as a
        sequence of ``(name, type)`` tuples where ``type`` must be a NumPy
        `scalar`_, or ``(name, type, converter)`` tuples where ``converter``
        is a function that converts the column string into the desired value.
    :param comment: The characters, or list of characters, that indicate the
        start of a single-line comment.
    :param encoding: The name of the encoding used to decode the file content.

    :raises ValueError: if ``columns`` contains a string column.

    :Examples:

    >>> from pypfilt.io import date_column, read_table
    >>> import numpy as np
    >>> import datetime
    >>> path = "input_data.ssv"
    >>> with open(path, 'w') as f:
    ...    _ = f.write('time value\\n')
    ...    _ = f.write('2020-01-01 1\\n')
    ...    _ = f.write('2020-01-02 3\\n')
    ...    _ = f.write('2020-01-03 5\\n')
    >>> columns = [date_column('time'), ('value', np.int_)]
    >>> data = read_table(path, columns)
    >>> isinstance(data['time'][0], datetime.datetime)
    True
    >>> observations = [{'time': row['time'], 'value': row['value']}
    ...                 for row in data]
    >>> # Remove the input file when it is no longer needed.
    >>> import os
    >>> os.remove(path)

    .. _scalar: https://numpy.org/doc/stable/reference/arrays.scalars.html

    .. _numpy.genfromtxt: https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
    """

    # Ensure there are no string fields.
    for column in columns:
        field_name = column[0]
        field_type = column[1]
        if isinstance(field_type, np.dtype):
            if h5py.check_string_dtype(field_type) is not None:
                msg_fmt = 'String column {} is not supported'
                raise ValueError(msg_fmt.format(field_name))

    # Read in the column names, and find the line where the data begins.
    skip_lines = 1
    with open(path, encoding=encoding) as f:
        file_cols = f.readline().strip().split()
        while len(file_cols) == 0 or file_cols[0].startswith(comment):
            file_cols = f.readline().strip().split()
            skip_lines += 1

    # Ensure all of the required columns are defined.
    need_columns = [col_tuple[0] for col_tuple in columns]
    for column_name in need_columns:
        if column_name not in file_cols:
            raise ValueError('Column "{}" not found in {}'.format(
                column_name, path))

    # Construct the list of column types and associate column conversion
    # functions with the index of that column in the file.
    converters = {}
    column_dtypes = []
    for ix, col_tuple in enumerate(columns):
        if len(col_tuple) == 2:
            column_dtypes.append(col_tuple)
        elif len(col_tuple) == 3:
            column_dtypes.append(col_tuple[:2])
            column_ix = file_cols.index(col_tuple[0])
            converters[column_ix] = col_tuple[2]

    # Determine the index of each required column in the file.
    read_columns = [file_cols.index(name) for name in need_columns]

    tbl = np.loadtxt(path, encoding=encoding,
                     skiprows=skip_lines, dtype=column_dtypes,
                     converters=converters, usecols=read_columns)

    # If the table only contains a single row, it will be represented as a
    # scalar value. So we need to convert it to an array with at least one
    # dimension.
    return np.atleast_1d(tbl)


def read_fields(time_scale, path, fields, comment='#', encoding='utf-8'):
    """
    Read data from a space-delimited text file with column headers defined in
    the first non-comment line.

    This is wrapper for :func:`read_table` that ensures time columns are
    identifiable.

    .. note:: Use :func:`~pypfilt.io.time_field` to identify columns that
       contain time values.
       See the example below.

    .. warning::

       This **does not handle string columns**.
       See :func:`~pypfilt.io.read_table` for a potential solution.

    :param time_scale: The simulation time scale.
    :param path: The path to the data file.
    :param fields: The columns to read from the data file, represented as a
        sequence of ``(name, type)`` tuples, where ``type`` must be a NumPy
        `scalar`_.
    :param comment: The characters, or list of characters, that indicate the
        start of a single-line comment.
    :param encoding: The name of the encoding used to decode the file content.

    :raises ValueError: if ``columns`` contains a string column.

    :Example:

    The following function reads a time series of floating-point values.

    .. code-block:: python

       import numpy as np
       import pypfilt.io

       def load_time_series(self, filename, time_scale):
           fields = [pypfilt.io.time_field('time'), ('value', np.float_)]
           return pypfilt.io.read_fields(time_scale, filename, fields)

    .. _scalar: https://numpy.org/doc/stable/reference/arrays.scalars.html
    """
    # Convert time fields into ``(name, type, converter)`` tuples for use with
    # read_table().
    columns = []
    for ix in range(len(fields)):
        # NOTE: either ``(name, type)`` or ``(name, type, shape)``.
        field_info = list(fields[ix])
        field_name = field_info[0]
        field_type = field_info[1]
        if len(field_info) != 2:
            msg_fmt = 'Field {} has non-scalar shape {}'
            raise ValueError(msg_fmt.format(field_name, field_info[2]))
        # Ensure there are no string fields.
        if isinstance(field_type, np.dtype):
            if h5py.check_string_dtype(field_type) is not None:
                msg_fmt = 'String column {} is not supported'
                raise ValueError(msg_fmt.format(field_name))
        if field_type == 'TIME':
            columns.append(time_scale.column(field_name))
        else:
            columns.append(fields[ix])

    table = read_table(path, columns, comment=comment, encoding=encoding)
    out_dtype = fields_dtype(time_scale, fields)
    return table.astype(out_dtype)


def date_column(name, fmt='%Y-%m-%d'):
    """
    Return a ``(name, type, converter)`` tuple that can be used with
    :func:`read_table` to convert a column into ``datetime.datetime`` values.

    .. note::

       Where dates are used for observation times, they should be represented
       as ``datetime.datetime`` values, not as ``datetime.date`` values.
       This is why this function returns a converter that returns
       ``datetime.datetime`` values.

    :param str name: The column name in the data file.
    :param str fmt: The date format used to parse the column values.
    """
    return (name, np.object_,
            lambda s: datetime.datetime.strptime(s, fmt))


def datetime_column(name, fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Return a ``(name, type, converter)`` tuple that can be used with
    :func:`read_table` to convert a column into ``datetime.datetime`` values.

    :param str name: The column name in the data file.
    :param str fmt: The datetime format used to parse the column values.
    """
    return (name, np.object_,
            lambda s: datetime.datetime.strptime(s, fmt))


def read_lookup_table(path, time, dtype='f8', comment='#', encoding='utf-8'):
    """
    Read time-indexed data from a space-delimited text file with column
    headers defined in the first non-comment line.

    :param path: The path to the data file.
    :param pypfilt.time.Time time: The time scale.
    :param dtype: The type of the lookup values.
    :param comment: The characters, or list of characters, that indicate the
        start of a single-line comment.
    :param encoding: The name of the encoding used to decode the file content.

    :Examples:

    >>> from pypfilt.io import read_lookup_table, lookup
    >>> from pypfilt.time import Datetime
    >>> import datetime
    >>> path = "input_data.ssv"
    >>> with open(path, 'w') as f:
    ...    _ = f.write('time value1 value2 value3\\n')
    ...    _ = f.write('2020-01-01 1.0 1.5 2.0\\n')
    >>> time = Datetime()
    >>> table = read_lookup_table(path, time)
    >>> isinstance(table['time'][0], datetime.datetime)
    True
    >>> when = datetime.datetime(2020, 1, 1)
    >>> values = lookup(table, when)
    >>> len(values.shape) == 1
    True
    >>> all(isinstance(value, float) for value in values)
    True
    >>> # Remove the input file when it is no longer needed.
    >>> import os
    >>> os.remove(path)
    """

    # Read in the column names, and find the line where the data begins.
    skip_lines = 1
    with open(path, encoding='utf-8') as f:
        cols = f.readline().strip().split()
        while len(cols) == 0 or cols[0].startswith(comment):
            cols = f.readline().strip().split()
            skip_lines += 1

    columns = [(name, dtype) for name in cols[1:]]
    columns.insert(0, time.column(cols[0]))

    tbl = read_table(path, columns)

    # NOTE: rename the first column to 'time', so that the cache can identify
    # this as a time-indexed table.
    col_names = list(tbl.dtype.names)
    col_names[0] = 'time'
    tbl.dtype.names = col_names

    # If the table only contains a single row, it will be represented as a
    # scalar value. So we need to convert it to an array with at least one
    # dimension.
    tbl = np.atleast_1d(tbl)

    if len(tbl) == 0:
        raise ValueError("File '{}' contains no rows".format(path))

    # Count all columns except for 'time'.
    num_value_cols = len(tbl[0]) - 1

    # Transform this table with rows (time, value1, value2, ...) into a table
    # with rows (time, [values]).
    row_pairs = [(row[0], tuple(row)[1:]) for row in tbl]

    # Ensure that time fields are identified in the data type metadata.
    new_fields = [time_field('time'), ('value', dtype, (num_value_cols,))]
    new_dtype = fields_dtype(time, new_fields)

    value_tbl = np.array(row_pairs, dtype=new_dtype)
    return value_tbl


def lookup_values_count(lookup_table):
    """
    Return the number of value columns in a lookup table.
    """
    # NOTE: ignore the first column, which contains the lookup time.
    return lookup_table.dtype['value'].shape[0]


def lookup(lookup_table, when):
    """
    Return the values associated with a specific time.
    """
    time_col = lookup_table.dtype.names[0]
    ixs = np.where(lookup_table[time_col] <= when)[0]
    if len(ixs) == 0:
        # No match, default to the earliest value.
        most_recent_row = 0
    else:
        # Otherwise use the most recent match.
        most_recent_row = ixs[-1]
    return lookup_table[most_recent_row]['value']


def lookup_times(lookup_table):
    """
    Return the times for which the lookup table contains values.
    """
    time_col = lookup_table.dtype.names[0]
    return lookup_table[time_col]


class Lookup:
    """
    Lookup tables provide a means of retrieving time-indexed quantities, which
    can be used to incorporate time-varying effects into simulation models and
    observation models.

    :param lookup_table: A data table, typically loaded with
        :func:`read_lookup_table`.
    """
    def __init__(self, lookup_table):
        self.__table = lookup_table

    def value_count(self):
        """Return the number of value columns in the lookup table."""
        return lookup_values_count(self.__table)

    def lookup(self, when):
        """Return the value(s) associated with a specific time."""
        return lookup(self.__table, when)

    def times(self):
        """Return the array of times for which values are defined."""
        return lookup_times(self.__table)

    def start(self):
        """Return the first time for which values are defined."""
        return self.times()[0]

    def end(self):
        """Return the final time for which values are defined."""
        return self.times()[-1]


def time_field(name):
    """
    Return a ``(name, type)`` tuple that identifies a field as containing time
    values.

    Use this function to define summary table fields that contain time values.

    :Examples:

    >>> import numpy as np
    >>> from pypfilt.io import time_field
    >>> fields = [time_field('time'), ('value', np.float_)]
    """
    return (name, 'TIME')


def string_field(name):
    """
    Return a ``(name, type)`` tuple that identifies a field as containing
    string values.

    Use this function to define summary table fields that contain string
    values.

    :Examples:

    >>> import numpy as np
    >>> from pypfilt.io import string_field
    >>> fields = [string_field('parameter_name'), ('value', np.float_)]
    """
    return (name, h5py.string_dtype())


def fields_dtype(time_scale, fields):
    """
    Return a NumPy data type (dtype) object that describes the provided data
    fields, and identifies fields that contain time values and string values.

    :param time_scale: The simulation time scale, or a simulation context.
    """
    # Accept either a time scale or a simulation context.
    from .time import Time
    if not isinstance(time_scale, Time):
        time_scale = time_scale.component['time']

    # Identify all columns that contain time or string values.
    # Note that we must not mutate the contents of ``fields``.
    time_native_dtype = time_scale.native_dtype()
    time_columns = []
    string_columns = []
    new_fields = []
    for ix in range(len(fields)):
        # NOTE: either ``(name, type)`` or ``(name, type, shape)``.
        field_info = list(fields[ix])
        field_name = field_info[0]
        field_type = field_info[1]
        if isinstance(field_type, str) and field_type == 'TIME':
            time_columns.append(field_name)
            # Replace the 'TIME' marker with the appropriate type.
            field_info[1] = time_native_dtype
            new_fields.append(tuple(field_info))
        else:
            if isinstance(field_type, np.dtype):
                if h5py.check_string_dtype(field_type) is not None:
                    string_columns.append(field_name)
            new_fields.append(fields[ix])

    # Record the time columns in the dtype metadata.
    metadata = {
        'time_columns': time_columns,
        'string_columns': string_columns,
    }
    dtype = np.dtype(new_fields, metadata=metadata)
    return dtype


def write_table(path, table, time_scale, columns=None, encoding='utf-8'):
    """
    Write a data table to a space-delimited text file with column headers.

    :param path: The path to the output file.
    :param table: The data table.
    :param time_scale: The simulation time scale.
    :type time_scale: pypfilt.time.Time
    :param columns: The subset of table columns to write to the output file.
    :param encoding: The name of the encoding used to decode the file content.

    .. warning:: This does not check whether string columns and time values
       contain whitespace.
    """
    if columns is None:
        columns = list(table.dtype.names)

    if table.dtype.metadata is None:
        time_column_ixs = []
    else:
        time_columns = table.dtype.metadata.get('time_columns')
        time_column_ixs = [
            ix for (ix, name) in enumerate(time_columns)
            if name in columns]

    row_format = ' '.join(['{}'] * len(columns)) + '\n'

    with open(path, 'w', encoding=encoding) as f:
        f.write('{}\n'.format(' '.join(columns)))

        # NOTE: we subset by columns inside this function to ensure that the
        # table's dtype metadata is preserved.
        for row in table[columns]:
            row = list(row.tolist())
            for ix in time_column_ixs:
                row[ix] = time_scale.to_unicode(row[ix])

            f.write(row_format.format(*row))


def load_dataset(time_scale, dataset):
    """
    Load a structured array from a HDF5 dataset, converting stored types into
    native types as necessary.

    :Examples:

    .. code-block:: python

       import h5py
       import pypfilt

       dataset_path = '/path/to/my/dataset'
       time_scale = pypfilt.Datetime()

       with h5py.File('output.hdf5', 'r') as f:
           dataset = f[dataset_path]
           table = pypfilt.io.load_dataset(time_scale, dataset)
    """

    # Make no modifications to unstructured arrays.
    table = dataset[()]
    if table.dtype.names is None:
        return table

    # Identify the columns that contain time values.
    encoded_columns = dataset.attrs.get('time_columns', [])
    time_columns = [name.decode() for name in encoded_columns]

    # Identify the columns that contain string values.
    encoded_columns = dataset.attrs.get('string_columns', [])
    string_columns = [name.decode() for name in encoded_columns]

    # Identify which columns need to be converted into a different type.
    conversion_table = native_time_converters(time_scale, table, time_columns)
    for column in string_columns:
        conversion_table[column] = (
            h5py.string_dtype(),
            lambda xs: np.array([x.decode() for x in xs]))

    # Construct the data type for the returned array.
    column_names = list(table.dtype.names)
    new_dtype = []
    for column in column_names:
        if column in conversion_table:
            (field_dtype, convert_fn) = conversion_table[column]
            field = (column, field_dtype)
        else:
            field = (column, table.dtype[column])
        new_dtype.append(field)

    metadata = {
        'time_columns': time_columns,
        'string_columns': string_columns,
    }
    new_dtype = np.dtype(new_dtype, metadata=metadata)

    # Construct the (empty) array and fill each column in turn.
    table_out = np.zeros(table.shape, dtype=new_dtype)
    for column in column_names:
        if column in conversion_table:
            (field_dtype, convert_fn) = conversion_table[column]
            table_out[column] = convert_fn(table[column])
        else:
            table_out[column] = table[column]

    return table_out


def save_dataset(time_scale, group, name, table, **kwargs):
    """
    Save a structured array as a HDF5 dataset, converting native types into
    stored types as necessary.

    :return: The HDF5 dataset.
    """
    logger = logging.getLogger(__name__)

    # Identify the columns that contain time values.
    if table.dtype.metadata is not None:
        msg_fmt = 'save_dataset(): {}/{} has dtype metadata'
        logger.debug(msg_fmt.format(group.name, name))
        time_columns = table.dtype.metadata.get('time_columns')
        string_columns = table.dtype.metadata.get('string_columns')
    else:
        msg_fmt = 'save_dataset(): {}/{} has no dtype metadata'
        logger.debug(msg_fmt.format(group.name, name))
        time_columns = None
        string_columns = None

    # Convert time values, if necessary.
    if time_columns is not None:
        table = ensure_stored_time(time_scale, table, time_columns)

    try:
        dataset = group.create_dataset(name, data=table, dtype=table.dtype,
                                       **kwargs)
    except Exception as e:
        msg_fmt = 'save_dataset(): could not save {}/{}'
        logger.error(msg_fmt.format(group.name, name))
        raise e

    # Save the time columns metadata in the dataset attributes.
    # We need to convert each column name into UTF-8 bytes.
    if time_columns is not None:
        encoded_columns = [column.encode() for column in time_columns]
        dataset.attrs['time_columns'] = np.array(encoded_columns)

    # Save the string columns metadata in the dataset attributes.
    # We need to convert each column name into UTF-8 bytes.
    if string_columns is not None:
        encoded_columns = [column.encode() for column in string_columns]
        dataset.attrs['string_columns'] = np.array(encoded_columns)

    return dataset


def native_time_converters(time_scale, table, columns):
    """
    Return a dictionary that maps column names to ``(dtype, convert_fn)``
    tuples.
    """
    if not isinstance(columns, list):
        columns = [columns]

    convert = __must_convert_to_native_time_cols(time_scale, table, columns)
    return {
        name: (time_scale.native_dtype(),
               lambda xs: [time_scale.from_dtype(x) for x in xs])
        for name in convert
    }


def ensure_stored_time(time_scale, table, columns):
    """
    Ensure that if ``column`` contains time values, that these values are in a
    format that can be saved to disk (e.g., in HDF5 format).

    :param time_scale: The simulation time scale.
    :type time_scale: pypfilt.time.Time
    :param table: The structured array that will be saved.
    :type table: numpy.ndarray
    :param columns: The name of the column(s) in the structured array that
        contain time values.
    :type column: Union[str, list[str]]
    """
    if not isinstance(columns, list):
        columns = [columns]

    convert = __must_convert_to_stored_time_cols(time_scale, table, columns)
    if len(convert) == 0:
        return table

    descr = list(table.dtype.names)
    col_ixs = []
    for ix in range(len(descr)):
        name = descr[ix]
        if name in convert:
            descr[ix] = time_scale.dtype(name)
            col_ixs.append(ix)
        else:
            descr[ix] = (name, table.dtype[name])

    rows = [list(row.tolist()) for row in table]
    for row in rows:
        for col_ix in col_ixs:
            row[col_ix] = time_scale.to_dtype(row[col_ix])
    rows = [tuple(row) for row in rows]
    return np.array(rows, dtype=descr)


def __equivalent_dtypes(dtype_1, dtype_2):
    """
    Return ``True`` if the two NumPy dtypes are equivalent.
    """
    return np.issubdtype(dtype_1, dtype_2) and np.issubdtype(dtype_2, dtype_1)


def __table_has_column_type(table, column, column_dtype):
    """
    Return ``True`` if a structured array contains a named column with the
    provided NumPy dtype.
    """
    return (table.dtype.names
            and column in table.dtype.names
            and __equivalent_dtypes(table.dtype[column].type,
                                    column_dtype))


def __must_convert_to_native_time_cols(time_scale, table, columns):
    """
    Return the list of columns that contain time values that are not in native
    format.
    """
    stored_dtype = np.dtype(time_scale.dtype('ignore')[1])
    native_dtype = np.dtype(time_scale.native_dtype())
    if __equivalent_dtypes(stored_dtype, native_dtype):
        return []
    return [column for column in columns
            if __table_has_column_type(table, column, stored_dtype)]


def __must_convert_to_stored_time_cols(time_scale, table, columns):
    """
    Return the list of columns that contain time values that are not in stored
    format.
    """
    stored_dtype = np.dtype(time_scale.dtype('ignore')[1])
    native_dtype = np.dtype(time_scale.native_dtype())
    if __equivalent_dtypes(stored_dtype, native_dtype):
        return []
    return [column for column in columns
            if __table_has_column_type(table, column, native_dtype)]


def ensure_directory_exists(path):
    """
    Ensure that `path` exists and is a directory.
    """
    logger = logging.getLogger(__name__)

    if not os.path.isdir(path):
        try:
            msg_fmt = 'Creating output directory {}'
            logger.info(msg_fmt.format(path))
            os.makedirs(path, exist_ok=True)
        except Exception:
            msg_fmt = 'Could not create output directory {}'
            logger.error(msg_fmt.format(path))
            raise
