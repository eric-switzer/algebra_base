def _set_type_axes(array, type, axis_names):
    """Sets the array.info['type'] and array.info[axes] metadata and does some
    checks.  Used in vect and mat constructors.
    """

    assert_info(array)

    if axis_names is None:
        axes = (None,)*array.ndim
    else:
        _check_axis_names(array, axis_names)
        axes = tuple(axis_names)

    array.info['type'] = type
    array.axes = axes


def _check_axis_names(array, axis_names=None):
    """Checks that axis names  sequence is valid for array."""

    if axis_names is None:
        axis_names = array.axes

    if len(axis_names) != array.ndim:
        raise ValueError("axis_names parameter must be a sequence of length "
                         "arr.ndim")
    else:
        for name in axis_names:
            if (not isinstance(name, str)) and (name is not None):
                raise TypeError("Invalid axis name.")


def _check_rows_cols(arr, row_axes=None, col_axes=None):
    """Check that rows and cols are valid for the matrix."""

    if row_axes is None and col_axes is None:
        row_axes = arr.info['rows']
        col_axes = arr.info['cols']

    # Both parameters had better be sequences of integers.
    for ind in row_axes:
        if ind not in range(arr.ndim):
            raise ValueError("Invalid row axes.")

    for ind in col_axes:
        if ind not in range(arr.ndim):
            raise ValueError("Invalid col axes")

    # Make sure each axis is spoken for.
    for ii in range(arr.ndim):
        if (ii not in row_axes) and (ii not in col_axes):
            raise ValueError("Every axis must be identified varying over "
                             "as the matrix row, column or both.")



