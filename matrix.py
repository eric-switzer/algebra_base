import sys
import base
import info_header
import warnings

class MatrixObject(base.AlgObject):
    """Multidimentional array interpreted as a matrix.

    This class gets most of its functionality from the numpy ndarray class.
    In addition it provides support for organizing it's data as a vector.
    This class comes in two flavours: `mat_array` and `mat_memmap`
    depending on whether the array is stored in memory or on disk.  The raw
    `mat` class is not a valid class by itself.

    To make the association between a multidimentional array and a matrix,
    each axis of the array must be identified as varying over either the
    rows or columns of a matrix.  For instance the shape of the array could
    be (3, 5, 7).  We could identify the first axis as a row axis and the
    second two as column axes in which case the matrix would have 3 rows
    and 35 colums.  We generally make the rows axes left of the columns and
    many algorithms assume this.  It is also possible for an axis to be
    identified as both a row and a column, in which case the matrix is
    block diagonal over that axis.  Generally, the block diagonal axes are
    the leftmost.

    Like `vect`s, mats have named axes, however 2 axes may have the same
    name as long as one is identified as a row axis and the other as a col
    axis.

    Parameters
    ----------
    input_array: InfoArray (for mat_array) or InfoMemmap (for
                  mat_memmap)
        Array to be converted to a vect.
    row_axes: tuple of ints
        Sequence contains the axis numbers of the array to identify as
        varying over the matrix rows. This sequence is stored in the
        `rows` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the rows.
    col_axis: tuple of ints
        Sequence contains the axis numbers of the array to identify as
        varying over the matrix columns. This sequence is stored in the
        `cols` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the cols.
    axis_names: tuple of strings, optional
        The sequence contains the name of each axis.  This sequence will be
        stored in the `axes` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the axis names.

    Attributes
    ----------
    axes: tuple of strings
        The names of each of the axes of the array.
    rows: tuple of ints
        Which of the array's axes to identify as varying over the matrix
        rows.
    cols: tuple of ints
        Which of the array's axes to identify as varying over the matrix
        columns.

    Notes
    -----
    Since much of the functionality provided by this class is only valid
    for a certain shape of the array, shape changing operations in
    general return an `InfoArray` or `InfoMemmap` as appropriate (except
    explicit assignment to mat.shape).

    The `axes`, `rows` and `cols` attributes are actually stored in the
    `InfoArray`'s info dictionary.  This is just an implementation detail.

    See Also
    --------
    vect_array, vect_memmap: Vector classes.
    make_mat: Function that casts any array as a matrix.
    InfoArray, InfoMemmap: Base classes that handle meta data.
    """

    __array_priority__ = 3.0

    def __new__(cls, input_array, row_axes=None, col_axes=None,
                axis_names=None):

        if not isinstance(input_array, cls.info_base):
            raise ValueError("Array to convert must be instance of " +
                             str(cls.info_base))

        obj = input_array.view(cls)

        if 'type' in obj.info:
            if axis_names is not None or \
               row_axes is not None or \
               col_axes is not None:

                warnings.warn("Initialization argument ignored. Requisite "
                              "metadata for matrix already exists. "
                              "Clear info dictionary if you want opposite "
                              "behaviour.")

            if obj.info['type'] != 'mat':
                raise ValueError("Meta data present is incompatible.")
            _check_axis_names(obj)
            _check_rows_cols(obj)
        else:
            if row_axes is None and \
               col_axes is None and \
               (input_array.ndim == 2):

                row_axes = (0,)
                col_axes = (1,)
            else:
                _check_rows_cols(input_array, row_axes, col_axes)
            _set_type_axes(obj, 'mat', axis_names)
            obj.rows = row_axes
            obj.cols = col_axes
        return obj

    def __setattr__(self, name, value):
        if name == 'axes':
            _check_axis_names(self, value)
            self.info['axes'] = value
        elif name == 'rows':
            for ind in value:
                if ind not in range(self.ndim):
                    raise ValueError("Invalid row axes.")

            self.info['rows'] = tuple(value)
        elif name == 'cols':
            for ind in value:
                if ind not in range(self.ndim):
                    raise ValueError("Invalid col axes.")

            self.info['cols'] = tuple(value)
        else:
            self.info_base.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'axes':
            return self.info['axes']
        elif name == 'rows':
            return self.info['rows']
        elif name == 'cols':
            return self.info['cols']
        else:
            # Since numpy uses __get_attribute__ not __getattr__, we should
            # raise an Attribute error.
            raise AttributeError("Attribute " + name + " not found.")

    def check_rows_cols(self):
        """Check that rows and cols are valid for the matrix.

        Raises an exception if the rows or columns are invalid.
        """
        _check_rows_cols(self)

    def assert_axes_ordered(self):
        """Enforces a specific ordering to the matrix row and column axis
        associations.
        """

        rows = self.info['rows']
        cols = self.info['cols']
        r_ind = len(rows) - 1
        c_ind = len(cols) - 1
        in_rows = False
        in_cols = True

        for axis in range(self.ndim-1, -1, -1):
            if cols[c_ind] == axis and in_cols and c_ind >= 0:
                c_ind -= 1
                continue
            elif in_cols:
                in_cols = False
                in_rows = True
            if rows[r_ind] == axis and in_rows and r_ind >= 0:
                r_ind -= 1
                continue
            elif in_rows:
                in_rows = False
            if rows[r_ind] == axis and cols[c_ind] == axis:
                r_ind -= 1
                c_ind -= 1
            else:
                raise NotImplementedError("Matrix row and column array axis"
                                          "linkage not ordered correctly.")

    def get_num_blocks(self, return_block_shape=False,
                       return_n_axes_diag=False):
        """Get the number of blocks in a block diagonal matrix."""

        shape = self.mat_shape()
        # Current algorithm assumes specific format.
        self.assert_axes_ordered()

        diag_axes = [ii for ii in range(self.ndim) if ii in self.rows
                     and ii in self.cols]
        num_blocks = sp.prod([self.shape[ii] for ii in diag_axes])

        if return_block_shape and return_n_axes_diag:
            return num_blocks, \
                   (shape[0]/num_blocks, shape[1]/num_blocks), \
                   len(diag_axes)

        elif return_block_shape:
            return num_blocks, (shape[0]/num_blocks, shape[1]/num_blocks)

        elif return_n_axes_diag:
            return num_blocks, len(diag_axes)
        else:
            return num_blocks

    def row_names(self):
        """Return the axis names that correspond to rows."""

        names = ()
        for axis_ind in self.rows:
            names = names + (self.axes[axis_ind],)

        return names

    def col_names(self):
        """Return the axis names that correspond to columns."""

        names = ()
        for axis_ind in self.cols:
            names = names + (self.axes[axis_ind],)

        return names

    def row_shape(self):
        """Return the shape of the array only including axes that correspond to
        rows."""

        shape = ()
        for axis_ind in self.rows:
            shape = shape + (self.shape[axis_ind],)

        return shape

    def col_shape(self):
        """Return the shape of the array only including axes that correspond to
        rows."""

        shape = ()
        for axis_ind in self.cols:
            shape = shape + (self.shape[axis_ind],)

        return shape

    def mat_shape(self):
        """Get the shape of the represented matrix."""

        self.check_rows_cols()
        _check_axis_names(self)
        nrows = 1

        for axis in self.rows:
            nrows *= self.shape[axis]

        ncols = 1

        for axis in self.cols:
            ncols *= self.shape[axis]

        return (nrows, ncols)

    def iter_blocks(self):
        """Returns an iterator over the blocks of a matrix."""

        # Build the iterator class.
        class iterator(object):
            def __init__(self, arr):
                self.arr = arr
                self.n_blocks, self.block_shape, self.n_axes_diag = \
                    arr.get_num_blocks(True, True)
                self.ii = 0

            def __iter__(self):
                return self

            def next(self):
                if self.ii >= self.n_blocks:
                    raise StopIteration()
                else:
                    # Find the indices for this block.
                    array_index = ()
                    tmp_block_num = self.ii
                    self.ii += 1
                    for jj in range(self.n_axes_diag - 1, -1, -1):
                        array_index = ((tmp_block_num % self.arr.shape[jj],)
                                       + array_index)

                        tmp_block_num = tmp_block_num//self.arr.shape[jj]

                    # Return the data.
                    return sp.reshape(self.arr[array_index], self.block_shape)

        # Initialize it and return it.
        return iterator(self)

    def _iter_row_col_index(self, row_or_col):
        """Implementation of iter_col_index and iter_row_index."""
        #
        # Build the iterator class.
        class iterator(object):
            def __init__(self, arr):
                self.ndim = arr.ndim
                self.ii = 0

                # Axes that we will iteration over.
                if row_or_col == 'col':
                    self.axes = list(arr.cols)
                    other_axes = arr.rows
                elif row_or_col == 'row':
                    self.axes = list(arr.rows)
                    other_axes = arr.cols
                else:
                    raise RunTimeError()

                # Do not iterate over axes that are shared with rows.
                for oth in other_axes:
                    if oth in self.axes:
                        self.axes.remove(oth)

                # Get the shape of these axes, as well as the total size.
                self.shape = ()
                self.size = 1
                for axis in self.axes:
                    self.shape += (arr.shape[axis],)
                    self.size *= arr.shape[axis]

            def __iter__(self):
                return self

            def next(self):
                inds = ()
                ii = self.ii
                self.ii += 1
                if ii >= self.size:
                    raise StopIteration()

                # The sequence that will eventually be used to subscript the
                # array.
                array_index = [slice(sys.maxint)] * self.ndim

                # Get the indices.  Loop through the axes backward.
                for jj in range(len(self.axes) - 1, -1, -1):
                    array_index[self.axes[jj]] = ii % self.shape[jj]
                    ii = ii//self.shape[jj]

                # Return the indices.
                return array_index

        # Initiallize and return iterator.
        return iterator(self)

    def iter_row_index(self):
        """Returns an iterator over row axes of the mat.

        This iterates over all the axes that are assotiated only with rows
        of the mat.  Any axis that is identified as both a column and a row is
        not iterated over.  The iterator returns an tuple that can subscript
        the mat (not a view of the mat).  This is useful when you have an
        operation that has to be applied uniformly to all the columns of a mat.

        Examples
        --------
        >>> for index in mat.iter_row_index():
        >>>     sub_arr = mat[index]
        >>>     sub_arr.shape == mat.col_shape()
        True
        """

        return self._iter_row_col_index('row')

    def iter_col_index(self):
        """Returns an iterator over column axes of the mat.

        This iterates over all the axes that are assotiated only with columns
        of the mat.  Any axis that is identified as both a column and a row is
        not iterated over.  The iterator returns an tuple that can subscript
        the mat (not a view of the mat).  This is useful when you have an
        operation that has to be applied uniformly to all the columns of a mat.

        Examples
        --------
        >>> for index in mat.iter_col_index():
        >>>     sub_arr = mat[index]
        >>>     sub_arr.shape == mat.row_shape()
        True
        """

        return self._iter_row_col_index('col')

    def mat_diag(self):
        """Get the daigonal elements of the matrix, as a vect object."""

        # Current algorithm assumes specific format.
        self.assert_axes_ordered()
        # We expect a square matrix
        shape = self.mat_shape()
        if shape[0] != shape[1]:
            raise NotImplementedError("Only works for square mats.")

        # output memory
        out = sp.empty((shape[0],))
        # Figure out how many axes are in both row and col (and therefore block
        # diagonal).
        n_blocks, block_shape = self.get_num_blocks(True, False)
        # For square matricies, n_blocks*block_shape[ii] == shape[ii].
        block_size = shape[0]//n_blocks

        # Transfer over the diagonals.
        for ii, mat_block in enumerate(self.iter_blocks()):
            out[ii*block_size:(ii+1)*block_size] = sp.diag(mat_block)

        # Now make this a vect object and transfer the relevant metadata.
        if self.row_shape() == self.col_shape():
            out.shape = self.row_shape()

        out = make_vect(out)

        if self.row_names() == self.col_names():
            out.axes = self.row_names()
            out.copy_axis_info(self)

        return out

    def expand(self):
        """Calculates expanded matrix in 2 dimensional form.

        Takes an arbitrary matrix and returns the expanded version of it,
        as matrix with internal array dimensions of shape(mat).  If the
        original matrix has efficiency from any block diagonal structure, this
        is lost in the returned matrix.
        """

        # XXX Obviouse improvement: Check if this matrix is already full
        # (ie not diagonal structure)
        # and if so, return a view.

        # Also verifies the validity of the matrix.
        shape = self.mat_shape()
        # Current algorithm assumes specific format.
        self.assert_axes_ordered()
        # Allocate memory.
        out_mat = sp.zeros(shape, dtype=self.dtype)
        out_mat = info_header.InfoArray(out_mat)
        out_mat = mat_array(out_mat)

        # Figure out how many axes are in both row and col (and therefore block
        # diagonal).
        n_blocks, block_shape = self.get_num_blocks(True, False)

        # Loop over the blocks and assign data.
        for ii, mat_block in enumerate(self.iter_blocks()):
            # Figure out where this block starts.
            row_start = ii*block_shape[0]
            col_start = ii*block_shape[1]
            out_mat[row_start:row_start + block_shape[0],
                    col_start:col_start + block_shape[1]] = mat_block

        return out_mat

    def mat_transpose(self):
        """Transpose the matrix.

        Returns an `mat` object with rows and columns exchanged.  The
        underlying array is not modified.

        Returns
        -------
        transposed_matrix: mat object
            A view of `self`, with different meta data such that the matrix is
            transposed.
        """

        # Make a copy of the info dictionary.
        info = dict(self.info)
        # Make a view of self.
        out = self.view()
        # Replace the info dictionary (which is a reference to self.info) with
        # the copy.
        out.info = info
        # Transpose the axes.
        out.cols = self.rows
        out.rows = self.cols
        return out

    def index_axis(self, axis, index):
        """Remove an axis by indexing with an integer.

        Returns a view of the `mat` with an axis removed by indexing it with
        the passed integer.  The output retains it's `mat` characteristics,
        which are updated to reflect the removal of the index.

        Parameters
        ----------
        axis: integer
            Axis to remove.
        index: integer
            Which entry to choose from `axis`.

        Returns
        -------
        out: mat or vect
            If the removal of the axis deplete either the rows or columns, a
            vect object is returned.  Otherwise a mat is returned.
        """
        # Suport negitive indicies.
        if axis < 0:
            axis = self.ndim + axis

        # Get copies of rows and colums ommitting the removed axis.  Also, any
        # indecies that are greater than axis, need to decremented.
        rows = list(self.rows)
        if axis in rows:
            rows.remove(axis)

        cols = list(self.cols)
        if axis in cols:
            cols.remove(axis)

        for ii in range(len(rows)):
            if rows[ii] > axis:
                rows[ii] -= 1

        for ii in range(len(cols)):
            if cols[ii] > axis:
                cols[ii] -= 1

        rows = tuple(rows)
        cols = tuple(cols)
        # New name attribute.
        names = self.axes[:axis] + self.axes[axis+1:]
        # Index the array.
        slice_index = [slice(None)] * self.ndim
        slice_index[axis] = index
        out = sp.asarray(self[slice_index])
        # Set the matrix meta data.
        if not rows or not cols:
            out = make_vect(out, axis_names=names)
        else:
            out = make_mat(out, axis_names=names, row_axes=rows, col_axes=cols)

        out.copy_axis_info(self)

        return out


def _mat_class_factory(base_class):
    """Internal class factory for making a matrix class that inherits from
    either InfoArray or InfoMemmap."""

    if (base_class is not info_header.InfoArray) and \
       (base_class is not info_header.InfoMemmap):
        raise TypeError("Matrices inherit from info arrays or info memmaps.")

    class mat_class(MatrixObject, base_class):
        __doc__ = MatrixObject.__doc__
        info_base = base_class

    return mat_class

mat_array = _mat_class_factory(info_header.InfoArray)
mat_array.__name__ = 'mat_array'
mat_memmap = _mat_class_factory(info_header.InfoMemmap)
mat_memmap.__name__ = 'mat_memmap'


def make_mat(array, row_axes=None, col_axes=None, axis_names=None):
    """Do what ever it takes to make a mat out of an array.

    Convert any class that can be converted to a mat (array, InfoArray,
    memmap, InfoMemmap) to the appropriate mat object (mat_array or
    mat_memmap).

    This convieiance function just simplifies the constructor heirarchy.
    Normally to get an mat out of an array, you would need to construct an
    intermediate InfoArray object.  This bypasses that step.

    Parameters
    ----------
    array: array_like
        Array to be converted to mat object (if possible).
    row_axes: tuple of ints
        Sequence contains the axis numbers of the array to identify as
        varying over the matrix rows. This sequence is stored in the
        `rows` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the rows.
    col_axis: tuple of ints
        Sequence contains the axis numbers of the array to identify as
        varying over the matrix columns. This sequence is stored in the
        `cols` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the cols.
    axis_names: tuple of strings, optional
        The sequence contains the name of each axis.  This sequence will be
        stored in the `axes` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the axis names.

    Returns
    -------
    mat_arr: mat_array or mat_memmap
        A view of `array` converted to a mat object.
    """
    if isinstance(array, sp.memmap):
        if not isinstance(array, info_header.InfoMemmap):
            array = info_header.InfoMemmap(array)
        return mat_memmap(array, row_axes, col_axes, axis_names)
    elif isinstance(array, sp.ndarray):
        if not isinstance(array, info_header.InfoArray):
            array = info_header.InfoArray(array)
        return mat_array(array, row_axes, col_axes, axis_names)
    else:
        raise TypeError("Object cannot be converted to a matrix.")



