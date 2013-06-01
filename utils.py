import numpy as np
import operator


def empty_like(obj):
    """Create a new algebra object with uninitialized data but otherwise the
    same as the passed object."""

    out = sp.empty_like(obj)
    return as_alg_like(out, obj)


def zeros_like(obj):
    """Create a new algebra object full of zeros but otherwise the same
    as the passed object."""

    out = sp.zeros_like(obj)
    return as_alg_like(out, obj)


def ones_like(obj):
    """Create a new algebra object full of zeros but otherwise the same
    as the passed object."""

    out = sp.ones_like(obj)
    return as_alg_like(out, obj)


def as_alg_like(array, obj):
    """Cast an array as an algebra object similar to the passed object.

    Parameters
    ----------
    array: numpy array
        Array to be cast
    obj: AlgObject
        Algebra object from which propertise should be copied.
    """

    if not isinstance(obj, AlgObject):
        raise TypeError("Object to mimic must be an `AlgObject`.")

    out = array
    out = InfoArray(out)
    out.info = dict(obj.info)

    if isinstance(obj, vect):
        out = make_vect(out)
    elif isinstance(obj, mat):
        out = make_mat(out)
    else:
        raise TypeError("Expected `obj` to be an algebra mat or vect.")

    return out


# TODO: These need scipy standard documentation.
def array_summary(array, testname, axes, meetall=False, identify_entries=True):
    """helper function for summarizing arrays
    meetall: prints those entries for which all values in the slice meet the
    criteria (normal behavior is print all entries where _any_ value in the
    slice meets the criteria
    identify_entries: prints entries meeting the criteria
    """
    total_matching = array.sum()

    if total_matching != 0:
        print testname + "s:"
        match_count = np.apply_over_axes(np.sum, array, axes)
        print match_count.flatten()
        if identify_entries:
            if meetall:
                arrayshape = array.shape
                subarray_size = reduce(operator.mul,
                                       [arrayshape[i] for i in axes])
                print "with all " + testname + "s: " + \
                      repr(np.where(match_count.flatten() == subarray_size))
            else:
                print "has " + testname + ": " + \
                      repr(np.where(match_count.flatten() != 0))
            print "total " + testname + "s: " + repr(total_matching)
        print "-" * 80
    else:
        print "There are no " + testname + " entries"


def compressed_array_summary(array, name, axes=[1, 2], extras=False):
    """print various summaries of arrays compressed along specified axes"""

    print "-" * 80
    print "array property summary for " + name + ":"
    array_summary(np.isnan(array), "nan", axes)
    array_summary(np.isinf(array), "inf", axes)
    array_summary((array == 0.), "zero", axes, meetall=True)
    array_summary((array < 0.), "negative", axes, identify_entries=False)

    if extras:
        sum_nu = np.apply_over_axes(np.sum, array, axes)
        min_nu = np.apply_over_axes(np.min, array, axes)
        max_nu = np.apply_over_axes(np.max, array, axes)
        print sum_nu.flatten()
        print min_nu.flatten()
        print max_nu.flatten()
    print ""


def cartesian(arrays, out=None):
    """
    SO: using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays. ~5x faster than itertools

    Parameters
    ----------
    arrays: list of array-like
        1-D arrays to form the cartesian product of.
    out: ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out: ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0: m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m: (j + 1) * m, 1:] = out[0: m, 1:]

    return out


def roll_zeropad(a, shift, axis=None):
    """
    SO: python-numpy-roll-with-padding
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a: array_like
        Input array.
    shift: int
        The number of places by which elements are shifted.
    axis: int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res: ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll    : Elements that roll off one end come back on the other.
    rollaxis: Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0:
        return a

    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))

        res = np.concatenate((a.take(np.arange(n-shift, n), axis),
                              zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)

    if reshape:
        return res.reshape(a.shape)
    else:
        return res

