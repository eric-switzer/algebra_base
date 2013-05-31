@_replace_write_header
def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=(1, 0), metafile=None):
    """Open a file and memory map it to an info_memmap object.

    This is similar to the numpy.lib.format.openmemmap() function but also
    deals with the meta data dictionary, which is read and written from a
    meta data file.

    The only extra argument over the numpy version is the meta data file name
    `metafile`.

    Parameters
    ----------
    metafile: str
        File name for which the `info` attribute of the returned info_memmap
        will be read from and written to. Default is None, where the it is
        assumed to be `filename` + ".meta".

    Returns
    -------
    marray: info_memmap
        The `info` is intialized as an empty dictionary if `mode` is 'w' or if
        the file corresponding to `metafile` does not exist.  The `metafile`
        attribute of marray is set to the `metafile` parameter unless `mode` is
        'r' or 'c' in which case it is set to None.
    """

    # Restrict to version (1,0) because we've only written write_header for
    # this version.
    if version != (1, 0):
        raise ValueError("Only version (1,0) is safe from this function.")

    # Memory map the data part.
    marray = npfor.open_memmap(filename, mode, dtype, shape, fortran_order,
                               version)

    # Get the file name for the meta data.
    if metafile is None:
        metafile = filename + '.meta'

    # Read the meta data if need be.
    if ('r' in mode or mode is 'c') and os.path.isfile(metafile):
        info_fid = open(metafile, 'r')
        try:
            infostring = info_fid.readline()
        finally:
            info_fid.close()
        info = safe_eval(infostring)
    else:
        info = {}

    # In read mode don't pass a metafile to protect the meta data.
    if mode is 'r' or mode is 'c':
        metafile = None

    marray = info_memmap(marray, info, metafile)

    return marray


def load(file, metafile=None):
    """Open a .npy file and load it into memory as an info_aray.

    Similar to the numpy.load function.  Does not support memory
    mapping (use open_memmap).

    Parameters
    ----------
    file: file handle or str
        .npy file or file name to read the array from.
    metafile: str
        File name for which the `info` attribute of the returned info_array
        will be read from. Default is None, where the it is
        assumed to be the file name associated with `file` with ".meta"
        appended. If the file does not exist, the info attribute is initialized
        to an empty dictionary.

    Returns
    -------
    iarray: info_array object
    """

    # Load the data from .npy format.
    array = sp.load(file)

    # Figure out what the filename for the meta data should be.
    if metafile is None:
        try:
            fname = file.name
        except AttributeError:
            fname = file
        metafile = fname + ".meta"

    # Read the meta data.
    if os.path.isfile(metafile):
        info_fid = open(metafile, 'r')
        try:
            infostring = info_fid.readline()
        finally:
            info_fid.close()
        info = safe_eval(infostring)
    else:
        info = {}

    # Construct the infor array.
    array = info_array(array, info)

    return array


@_replace_write_header
def save(file, iarray, metafile=None, version=(1, 0)):
    """Save a info array to a .npy file and a metadata file.

    Similar to the numpy.save function.

    Parameters
    ----------
    file: file handle or str
        File or file name to write the array to in .npy format.
    iarray: info_array object or array with similar interface
        Array to be written to file with meta data.
    metafile: str
        File name for the meta data.  The `info` attribute of `iarray` will be
        written here. Default is None, where the it is
        assumed to be the file name associated with `file` with ".meta"
        appended.
    """

    # Restrict to version (1,0) because we've only written write_header for
    # this version.
    if version != (1, 0):
        raise ValueError("Only version (1,0) is safe from this function.")

    # Make sure that the meta data will be representable as a string.
    infostring = repr(iarray.info)
    try:
        safe_eval(infostring)
    except SyntaxError:
        raise ce.DataError("Array info not representable as a string.")

    # Save the array in .npy format.
    if isinstance(file, basestring):
        fid = open(file, "wb")
    else:
        fid = file

    npfor.write_array(fid, iarray, version=version)

    # Figure out what the filename for the meta data should be.
    if metafile is None:
        try:
            fname = file.name
        except AttributeError:
            fname = file
        metafile = fname + ".meta"

    # Save the meta data.
    info_fid = open(metafile, 'w')
    try:
        info_fid.write(infostring)
    finally:
        info_fid.close()


def save_h5(h5obj, path, iarray):
    """Store the info array in an hdf5 file.

    Parameters
    ----------
    h5obj: h5py File or Group object
        File to which the info array will be written.
    path: string
        Path within `h5obj` to write the array.
    iarray: info_array
        info_array to write.
    """

    # TODO: Allow `h5obj` to be a string with a path to a new file to be
    # created (and closed at the end). Acctually, this would require us to
    # import h5py, which we don't want to do (could do it locally).
    data = h5obj.create_dataset(path, iarray.shape, iarray.dtype)
    data[:] = iarray[:]

    for key, value in iarray.info.iteritems():
        data.attrs[key] = repr(value)


def load_h5(h5obj, path):
    """Load an info array from an hdf5 file.

    Parameters
    ----------
    h5obj: h5py File or Group object
        File from which the info array will be read from.
    path: string
        Path within `h5obj` to read the array.

    Returns
    -------
    iarray: info_array
        Array loaded from file.
    """

    # TODO:  Allow `h5obj` to be a string with a path to a file to be opened
    # and then closed.
    data = h5obj[path]
    iarray = np.empty(data.shape, data.dtype)
    iarray[:] = data[:]
    info = {}

    for key, value in data.attrs.iteritems():
        info[key] = safe_eval(value)

    iarray = info_array(iarray, info)

    return iarray

