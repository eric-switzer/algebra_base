def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    This version of write array header has been modified to align the start of
    the array data with the 4096 bytes, corresponding to the page size of most
    systems.  This is so the npy files can be easily memmaped.

    Parameters
    ----------
    fp: filelike object
    d: dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    import struct
    header = ["{"]

    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))

    header.append("}")
    header = "".join(header)

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a 16-byte
    # boundary.  Hopefully, some system, possibly memory-mapping, can take
    # advantage of our premature optimization.
    # 1 for the newline
    current_header_len = npfor.MAGIC_LEN + 2 + len(header) + 1
    topad = 4096 - (current_header_len % 4096)
    header = '%s%s\n' % (header, ' '*topad)

    if len(header) >= (256*256):
        raise ValueError("header does not fit inside %s bytes" % (256*256))

    header_len_str = struct.pack('<H', len(header))
    fp.write(header_len_str)
    fp.write(header)


def _replace_write_header(f):
    """Wrap functions such that np.lib.format.write_array_header_1_0 is
    replaced by the local version, but only for the one function call."""
    @wraps(f)
    def wrapper(*args, **kwds):
        # Replace the header writer in the format module.
        tmp_write_header = npfor.write_array_header_1_0
        npfor.write_array_header_1_0 = write_array_header_1_0
        # Evaluate the function.
        try:
            result = f(*args, **kwds)
        finally:
            # Restore the header.
            npfor.write_array_header_1_0 = tmp_write_header
        return result

    return wrapper


class info_array(sp.ndarray):
    """A standard numpy ndarray object with a dictionary for holding
    extra info.

    This class should work exactly the same as a numpy ndarray object but has
    and attribute named info, which is a dictionary.  This class performs basic
    meta data handling for the higher lever classes that subclass this one:
    mat_array and vect_array.

    Parameters
    ----------
    input_array: array like
        Array to converted to an info_array.  The info_array will be a
        view to the input array.
    info: dictionary
        Dictionary to be set as the `info` attribute (default is None, which
        implies create a new empty dictionary).

    Attributes
    ----------
    info: dictionary
        Holds any meta data associated with this array.  All items should be
        easily represented as a string so they can be written to and read from
        file.  Setting this attribute is generally not safe, but modifying it
        is.

    See Also
    --------
    info_memmap: Analogous class to this one with data stored on disk.
    vect_array: Vector object based on this class.
    mat_array: Matrix object based on this class.

    Notes
    -----
    All new from template array creation operations make a copy of the metadata
    not a reference.  To get a reference you need to explicitly make a call to
    the `view` function.

    See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html for more
    information
    """

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance.
        # We first cast to be our class type.
        obj = sp.asarray(input_array).view(cls)

        # Add the new attribute to the created instance.
        if info is not None:
            obj.info = info

        # Finally, we must return the newly created object.
        return obj

    def __array_finalize__(self, obj):
        # `info` is a reference to the origional only for an explicit call to
        # self.view() (view casting).  Otherwise we copy to protect the data.
        self.info = dict(getattr(obj, 'info', {}))

    def view(self, *args):
        """Return a numpy view of self.

        This is mostly the same as the numpy version of this method, but it
        also makes the view's `info` attribute a reference to `self`s (where
        applicable).

        See Also
        --------
        np.ndarray.view
        """
        # Create the normal view.
        out = sp.ndarray.view(self, *args)

        # If it's info_array, replace the copy of the info attribute with a
        # reference (they will share metadata).
        if isinstance(out, info_array):
            out.info = self.info

        return out


class info_memmap(sp.memmap):
    """A standard numpy memmap object with a dictionary for holding extra info.

    This class should work exactly the same as a numpy memmap object but has an
    attribute named info, which is a dictionary. This class performs basic
    meta data handling for the higher lever classes that subclass this one:
    mat_memmap and vect_memmap.  This array is written to file at the same time
    that the memmap is flushed.

    Parameters
    ----------
    marray: numpy.memmap
        Array to be converted to an info_memmap.  The info_memmap will be a
        view to the input array.
    info: dictionary
        Dictionary to be set as the `info` attribute (default is None, which
        implies create a new empty dictionary).
    metafile: str
        filename to write the metadata to.  In some
        versions of numpy, the metadata will be written to file even if the
        memmap is in read only mode.  To avoid this pass metafile=None, which
        prevents the metadata from being stored on disk at all.

    Attributes
    ----------
    info: dictionary
        Holds any meta data associated with this array.  All items should be
        easily represented as a string so they can be written to and read from
        file. Setting this attribute is generally not safe, but modifying it
        is.
    metafile: str
        filename where the metadata is written to.  `info` is written to this
        file whenever the `flush` method is called (which includes deletion of
        the object).  This can happen even if the memmap was opened in 'r'
        mode.  Set to None if you wish to protect the data on file.

    See Also
    --------
    info_array: Similar class with data stored in memory.
    vect_memmap: Vector object based on this class.
    mat_memmap: Matrix object based on this class.
    open_memmap: Open a file on disk as an info_memmap.

    Notes
    -----
    All new from template array creation operations make a copy of the metadata
    not a reference.  To get a reference you need to explicitly make a call to
    the `view` function.  Also, the metafile is set to None on new from
    template oberations.

    See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html for more
    information
    """

    def __new__(cls, marray, info=None, metafile=None):
        # Input array is an already formed ndarray instance.
        # We first cast to be our class type.
        if not isinstance(marray, sp.memmap):
            raise TypeError("info_memmaps can only be initialized off of "
                            "numpy memmaps.")

        obj = marray.view(cls)

        # Add the new attribute to the created instance.
        if info is None:
            info = {}

        obj.info = info
        obj.metafile = metafile

        # Finally, we must return the newly created object.
        return obj

    def __array_finalize__(self, obj):
        sp.memmap.__array_finalize__(self, obj)

        # Info is a reference to the origional for views.
        self.info = dict(getattr(obj, 'info', {}))

        # Do not copy the metafile attribute, new arrays will clobber the data.
        # This attribute is only copied on an explicit view() call.
        self.metafile = None

    def view(self, *args):
        """Return a numpy view of self.

        This is mostly the same as the numpy version of this method, but it
        also makes the view's `info` attribute a reference to `self`s (where
        applicable).

        See Also
        --------
        np.ndarray.view
        """
        # Create the normal view.
        out = sp.memmap.view(self, *args)

        # If it's info_array, replace the copy of the info attribute with a
        # reference (they will share metadata).
        if isinstance(out, info_memmap):
            out.info = self.info
            out.metafile = self.metafile

        return out

    def flush(self):
        """Flush changes to disk.

        This method saves the info dictionary to metafile and then calls the
        flush method from the numpy memmap.
        """
        # Write the info dictionary to disk.
        self._info_flush()

        # Now flush the actual memmap.
        sp.memmap.flush(self)

    def _info_flush(self):
        """Write the info array to disk only."""
        # Prior to numpy 1.5, we can't get the mode, so just assume we are
        # allowed to write
        mode = getattr(self, 'mode', 'r+')
        if ('+' in mode or 'w' in mode) and self.metafile is not None:
            # Convert info dictionary to a pretty string.
            infostring = repr(self.info)

            try:
                safe_eval(infostring)
            except SyntaxError:
                raise ce.DataError("Array info not representable as a string.")

            # Save the meta data.
            info_fid = open(self.metafile, 'w')

            try:
                info_fid.write(infostring)
            finally:
                info_fid.close()

    def __del__(self):
        self._info_flush()
        sp.memmap.__del__(self)

    def __deepcopy__(self, copy):
        """Not implemented, raises an exception."""
        raise NotImeplementedError("Deep copy won't work.")


def assert_info(array):
    """Check if passed array is an info_array or info_memmap.

    Raises a ValueError if check fails.
    """
    if not (isinstance(array, info_array) or isinstance(array, info_memmap)):
        raise TypeError("Array is not an algebra.info_array or "
                        "algebra.info_memmap.")
