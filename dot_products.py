def dot(arr1, arr2, check_inner_axes=True):
    """Perform matrix multiplication."""

    shape1 = arr1.mat_shape()
    shape2 = arr2.mat_shape()

    if shape1[-1] != shape2[0]:
        raise ValueError("Matrix dimensions incompatible for matrix "
                         "multiplication.")

    # Matrix-vector product case.
    if len(shape1) == 2 and len(shape2) == 1:
        # Strict axis checking has been requested, make sure that the axis
        # number, lengths and names of the input vector are equal to the
        # column axis names of the input matrix.
        if check_inner_axes:
            if arr2.ndim != len(arr1.cols):
                raise ce.DataError("Matrix column axis number are not the "
                                   "same as vector ndim and strict checking "
                                   "has been requested.")
            for ii, name in enumerate(arr2.axes):
                if arr1.shape[arr1.cols[ii]] != arr2.shape[ii]:
                    raise ce.DataError("Matrix column axis lens are not the "
                                       "same as vector axis lens and strict "
                                       "checking has been requested.")
                if name != arr1.axes[arr1.cols[ii]]:
                    raise ce.DataError("Matrix column axis names are not the "
                                       "same as vector axes names and strict "
                                       "checking has been requested.")

        # Figure out what the output vector is going to look like.
        out_shape = [arr1.shape[ii] for ii in range(arr1.ndim)
                     if ii in arr1.info['rows']]
        out_names = [arr1.info['axes'][ii] for ii in range(arr1.ndim)
                     if ii in arr1.info['rows']]

        out_vect = sp.empty(out_shape)
        out_vect = make_vect(out_vect, out_names)
        n_blocks, block_shape = arr1.get_num_blocks(return_block_shape=True)
        # Make flattened veiws for the acctual matrix algebra.
        out_flat = out_vect.flat_view()
        arr2_flat = arr2.flat_view()

        for ii, block in enumerate(arr1.iter_blocks()):
            out_flat[ii*block_shape[0]:(ii+1)*block_shape[0]] = \
                sp.dot(block, arr2_flat[ii*block_shape[1]:
                                        (ii+1)*block_shape[1]])

        return out_vect
    else:
        raise NotImplementedError("Matrix-matrix multiplication has not been "
                                  "Implemented yet.")

def partial_dot(left, right):
    """Perform matrix multiplication on some subset of the axes.

    This is similar to a numpy `tensordot` but it is aware of the matrix and
    vector nature of the inputs and returns appropriate objects.  It decides
    which axes to 'dot' based on the axis names.

    If a `vect` is passed, it is treated as `mat` with one row if it's the
    first arguments and a matrix with one column if it's the second.  If the
    output matrix has either only a single row or a single column, it is cast
    as a `vect`.

    This function can properly deal with block diagonal structure and axes
    sorted in any order.

    The axes in the output array as sorted such in order of block diagonal axes
    then row-only axes then col-only axes.

    Parameters
    ----------
    left: mat or vect
    right: mat or vect

    Returns
    -------
    out: mat or vect
        Tensor product of `left` and `right`, with any named axes appearing in
        both `left`'s columns and `right`'s rows contracted.
    """

    # Figure out what kind of object the inputs are.
    msg = "Inputs must be either mat or vect objects."
    if isinstance(left, mat):
        left_rows = list(left.rows)
        left_cols = list(left.cols)
    elif isinstance(left, vect):
        left_rows = []
        left_cols = range(left.ndim)
    else:
        raise TypeError(msg)

    if isinstance(right, mat):
        right_rows = list(right.rows)
        right_cols = list(right.cols)
    elif isinstance(right, vect):
        right_rows = range(right.ndim)
        right_cols = []
    else:
        raise TypeError(msg)

    # Find axes that are block diagonal make copies of the rows and cols that
    # ommits the block diagonal ones.
    left_cols_only = list(left_cols)
    left_rows_only = list(left_rows)
    right_cols_only = list(right_cols)
    right_rows_only = list(right_rows)
    left_diag = []
    left_diag_names = []
    left_col_only_names = []
    for axis in left_cols:
        if axis in left_rows:
            left_diag.append(axis)
            left_diag_names.append(left.axes[axis])
            left_rows_only.remove(axis)
            left_cols_only.remove(axis)
        else:
            left_col_only_names.append(left.axes[axis])

    right_diag = []
    right_diag_names = []
    right_row_only_names = []
    for axis in list(right_rows):
        if axis in right_cols:
            right_diag.append(axis)
            right_diag_names.append(right.axes[axis])
            right_rows_only.remove(axis)
            right_cols_only.remove(axis)
        else:
            right_row_only_names.append(right.axes[axis])

    # Divide all axes into groups based on what we are going to do with them.
    # To not be dotted.
    left_notdot = []

    # Block diagonal axis to not dot.
    left_notdot_diag = []

    # To be dotted with a normal axis.
    left_todot = []
    right_todot = []

    # To be dotted with a block diagonal axis.
    left_todot_with_diag = []
    right_todot_with_diag = []

    # Block diagonal axes to be dotted with a normal axis.
    left_todot_diag = []
    right_todot_diag = []

    # Block diagonal axes to be dotted with a block diagonal axis.
    left_todot_diag_diag = []
    right_todot_diag_diag = []

    for axis in left_cols:
        axis_name = left.axes[axis]
        if axis_name in left_col_only_names \
           axis_name not in right_row_only_names \
           axis_name not in right_diag_names:

            left_notdot.append(axis)

        elif axis_name in left_diag_names and \
             axis_name not in right_row_only_names and \
             axis_name not in right_diag_names:

            left_notdot_diag.append(axis)

        elif axis_name in left_col_only_names and \
             axis_name in right_row_only_names:

            left_todot.append(axis)
            right_todot.append(right.axes.index(axis_name))

        elif axis_name in left_diag_names and \
             axis_name in right_row_only_names:

            left_todot_diag.append(axis)
            right_todot_with_diag.append(right.axes.index(axis_name))

        elif axis_name in left_col_only_names and \
             axis_name in right_diag_names:

            left_todot_with_diag.append(axis)
            right_todot_diag.append(right.axes.index(axis_name))

        elif axis_name in left_diag_names and \
             axis_name in right_diag_names:

            left_todot_diag_diag.append(axis)
            right_todot_diag_diag.append(right.axes.index(axis_name))

    right_notdot = list(set(right_rows_only) - set(right_todot)
                        - set(right_todot_with_diag))

    right_notdot.sort()

    right_notdot_diag = list(set(right_diag) - set(right_todot_diag)
                             - set(right_todot_diag_diag))

    right_notdot_diag.sort()

    # Need shapes and names for all of these.
    left_notdot_shape = [left.shape[axis] for axis in left_notdot]
    left_notdot_names = [left.axes[axis] for axis in left_notdot]
    left_notdot_diag_shape = [left.shape[axis] for axis in left_notdot_diag]
    left_notdot_diag_names = [left.axes[axis] for axis in left_notdot_diag]
    left_todot_shape = [left.shape[axis] for axis in left_todot]
    left_todot_names = [left.axes[axis] for axis in left_todot]

    left_todot_with_diag_shape = [left.shape[axis] for axis in
                                  left_todot_with_diag]

    left_todot_with_diag_names = [left.axes[axis] for axis in
                                  left_todot_with_diag]

    left_todot_diag_shape = [left.shape[axis] for axis in left_todot_diag]
    left_todot_diag_names = [left.axes[axis] for axis in left_todot_diag]

    left_todot_diag_diag_shape = [left.shape[axis] for axis in
                                  left_todot_diag_diag]

    left_todot_diag_diag_names = [left.axes[axis] for axis in
                                  left_todot_diag_diag]

    left_rows_only_shape = [left.shape[axis] for axis in left_rows_only]
    left_rows_only_names = [left.axes[axis] for axis in left_rows_only]
    right_notdot_shape = [right.shape[axis] for axis in right_notdot]
    right_notdot_names = [right.axes[axis] for axis in right_notdot]

    right_notdot_diag_shape = [right.shape[axis]
                               for axis in right_notdot_diag]

    right_notdot_diag_names = [right.axes[axis] for axis in right_notdot_diag]
    right_todot_shape = [right.shape[axis] for axis in right_todot]
    right_todot_names = [right.axes[axis] for axis in right_todot]

    right_todot_with_diag_shape = [right.shape[axis] for axis in
                                   right_todot_with_diag]

    right_todot_with_diag_names = [right.axes[axis] for axis in
                                   right_todot_with_diag]

    right_todot_diag_shape = [right.shape[axis] for axis in right_todot_diag]
    right_todot_diag_names = [right.axes[axis] for axis in right_todot_diag]

    right_todot_diag_diag_shape = [right.shape[axis] for axis in
                                   right_todot_diag_diag]

    right_todot_diag_diag_names = [right.axes[axis] for axis in
                                   right_todot_diag_diag]

    right_cols_only_shape = [right.shape[axis] for axis in right_cols_only]
    right_cols_only_names = [right.axes[axis] for axis in right_cols_only]

    # Figure out the shape, names, rows and cols of the output.
    out_shape = (left_notdot_diag_shape + left_todot_diag_diag_shape
                 + right_notdot_diag_shape + left_todot_diag_shape
                 + left_rows_only_shape + right_notdot_shape
                 + left_notdot_shape + right_todot_diag_shape
                 + right_cols_only_shape)

    out_names = (left_notdot_diag_names + left_todot_diag_diag_names
                 + right_notdot_diag_names + left_todot_diag_names
                 + left_rows_only_names + right_notdot_names
                 + left_notdot_names + right_todot_diag_names
                 + right_cols_only_names)

    # First add the block diagonal axes as both rows and columns.
    out_rows = range(len(left_notdot_diag) + len(left_todot_diag_diag)
                     + len(right_notdot_diag))

    out_cols = list(out_rows)

    # Now add the others.
    out_rows += range(len(out_rows), len(out_rows) + len(left_todot_diag)
                      + len(left_rows_only) + len(right_notdot))
    out_cols += range(len(out_rows), len(out_shape))

    # Output data type.
    # This is no good because it crashes for length 0 arrays.
    #out_dtype = (left.flat[[0]] * right.flat[[0]]).dtype
    # There are functions that do this in higher versions of numpy.
    out_dtype = sp.dtype(float)

    # Allowcate memory.
    out = sp.empty(out_shape, dtype=out_dtype)

    # All the block diagonal axes will be treated together. Get the global
    # shape of them.
    all_diag_shape = (left_notdot_diag_shape + left_todot_diag_diag_shape
                      + right_notdot_diag_shape + left_todot_diag_shape
                      + right_todot_diag_shape)

    n_diag_axes = len(all_diag_shape)
    all_diag_size = 1

    for s in all_diag_shape:
        all_diag_size *= s

    # Each of these block diagonal axes are associated with different axes in
    # the input and output arrays.  Figure out the associations for each of
    # them.  These arrays are the length of the number of diagonal axes and
    # each entry refers to an axis in that array.  If the axis does not apply
    # to that array, put None.
    out_diag_inds = []
    left_diag_inds = []
    right_diag_inds = []
    tmp_n_out_axes_passed = 0
    for ii in range(len(left_notdot_diag)):
        left_diag_inds.append(left_notdot_diag[ii])
        right_diag_inds.append(None)
        out_diag_inds.append(ii)

    tmp_n_out_axes_passed += len(left_notdot_diag)
    for ii in range(len(left_todot_diag_diag)):
        left_diag_inds.append(left_todot_diag_diag[ii])
        right_diag_inds.append(right_todot_diag_diag[ii])
        out_diag_inds.append(ii + tmp_n_out_axes_passed)

    tmp_n_out_axes_passed += len(left_todot_diag_diag)
    for ii in range(len(right_notdot_diag)):
        left_diag_inds.append(None)
        right_diag_inds.append(right_notdot_diag[ii])
        out_diag_inds.append(ii + tmp_n_out_axes_passed)

    tmp_n_out_axes_passed += len(right_notdot_diag)
    for ii in range(len(left_todot_diag)):
        left_diag_inds.append(left_todot_diag[ii])
        right_diag_inds.append(right_todot_with_diag[ii])
        out_diag_inds.append(ii + tmp_n_out_axes_passed)

    tmp_n_out_axes_passed += (len(left_todot_diag) + len(left_rows_only)
                              + len(right_notdot) + len(left_notdot))

    for ii in range(len(right_todot_diag)):
        left_diag_inds.append(left_todot_with_diag[ii])
        right_diag_inds.append(right_todot_diag[ii])
        out_diag_inds.append(ii + tmp_n_out_axes_passed)

    # Once we index all the diagonal axes, the ones we want to dot will be in
    # the wrong place.  Find the location of the axes we'll need in the new
    # array.
    left_sliced_rows_only = list(left_rows_only)
    left_sliced_notdot = list(left_notdot)
    left_sliced_todot = list(left_todot)
    for diag_axis in left_diag_inds:
        if diag_axis is not None:
            for ii in range(len(left_rows_only)):
                if diag_axis < left_rows_only[ii]:
                    left_sliced_rows_only[ii] -= 1
            for ii in range(len(left_notdot)):
                if diag_axis < left_notdot[ii]:
                    left_sliced_notdot[ii] -= 1
            for ii in range(len(left_todot)):
                if diag_axis < left_todot[ii]:
                    left_sliced_todot[ii] -= 1

    right_sliced_cols_only = list(right_cols_only)
    right_sliced_notdot = list(right_notdot)
    right_sliced_todot = list(right_todot)
    for diag_axis in right_diag_inds:
        if diag_axis is not None:
            for ii in range(len(right_cols_only)):
                if diag_axis < right_cols_only[ii]:
                    right_sliced_cols_only[ii] -= 1
            for ii in range(len(right_notdot)):
                if diag_axis < right_notdot[ii]:
                    right_sliced_notdot[ii] -= 1
            for ii in range(len(right_todot)):
                if diag_axis < right_todot[ii]:
                    right_sliced_todot[ii] -= 1

    # Once we slice the arrays to get one block, we will permute the axes to be
    # in the proper order for dotting.
    left_sliced_permute = (left_sliced_rows_only + left_sliced_notdot
                           + left_sliced_todot)

    right_sliced_permute = (right_sliced_todot + right_sliced_notdot
                            + right_sliced_cols_only)

    # Then we'll reshape both into 2D arrays (matricies).
    left_sliced_reshape = (sp.prod(left_rows_only_shape + left_notdot_shape),
                           sp.prod(left_todot_shape))

    right_sliced_reshape = (sp.prod(right_todot_shape),
                            sp.prod(right_notdot_shape +
                                    right_cols_only_shape))

    # After the dot, we will neet to reshape back.
    out_sliced_reshape = tuple(left_rows_only_shape + left_notdot_shape
                               + right_notdot_shape + right_cols_only_shape)

    # And finally we will need to permute the out axes.
    out_sliced_permute = range(len(left_rows_only))

    out_sliced_permute += range(len(left_rows_only) + len(left_notdot_shape),
                                len(left_rows_only) + len(left_notdot_shape)
                                + len(right_notdot))

    out_sliced_permute += range(len(left_rows_only), len(left_rows_only)
                                + len(left_notdot))

    out_sliced_permute += range(len(out_sliced_reshape) - len(right_cols_only),
                                len(out_sliced_reshape))

    # Create an index for each of left, right and out.
    left_slice = [slice(None)] * left.ndim
    right_slice = [slice(None)] * right.ndim
    out_slice = [slice(None)] * out.ndim

    # Flags for corner cases of 0D arrays.
    left_scalar_flag = False
    right_scalar_flag = False

    # Now we loop over all the block diagonal axes.
    for ii in xrange(all_diag_size):
        # Figure out exactly which blocks we are dealing with.
        tmp_ii = ii
        for kk in xrange(n_diag_axes - 1, -1, -1):
            this_index = tmp_ii % all_diag_shape[kk]
            out_slice[out_diag_inds[kk]] = this_index
            if left_diag_inds[kk] is not None:
                left_slice[left_diag_inds[kk]] = this_index
            if right_diag_inds[kk] is not None:
                right_slice[right_diag_inds[kk]] = this_index
            tmp_ii = tmp_ii // all_diag_shape[kk]

        this_left = left[tuple(left_slice)]
        this_right = right[tuple(right_slice)]

        # Permute and reshape the axes so the fast dot function can be used.
        # Corner case, if this_left or this_right are scalars (0D arrays).
        # TODO: Really these flags can be set outside the loop with a few more
        # lines of code.
        if ii == 0:
            if this_left.ndim == 0:
                left_scalar_flag = True
            if this_left.ndim == 0 and this_right.ndim == 0:
                right_scalar_flag = True

        if not left_scalar_flag:
            this_left = this_left.transpose(left_sliced_permute)
            this_left = sp.reshape(this_left, left_sliced_reshape)

        if not right_scalar_flag:
            this_right = this_right.transpose(right_sliced_permute)
            this_right = sp.reshape(this_right, right_sliced_reshape)

        # Dot them.
        if left_scalar_flag or right_scalar_flag:
            this_out = this_left * this_right
        else:
            this_out = sp.dot(this_left, this_right)

        # Reshape, permute and copy to the output.
        if not (left_scalar_flag and right_scalar_flag):
            this_out.shape = out_sliced_reshape
            this_out = this_out.transpose(out_sliced_permute)

        out[tuple(out_slice)] = this_out

    # XXX There is a bug where this crashes for certain cases if these lines
    # are before the loop.
    if not out_rows or not out_cols:
        out = make_vect(out, out_names)
    else:
        out = make_mat(out, axis_names=out_names, row_axes=out_rows,
                       col_axes=out_cols)

    return out

