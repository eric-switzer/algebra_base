import scipy as sp
import base
import info_header
import warnings
import helpers


class VectorObject(base.AlgObject):
    """Multidimentional array interpreted as a vector.

    This class gets most of its functionality from the numpy ndarray class.
    In addition it provides support for orgainizing its data as a vector.
    This class comes in two flavours: `vect_array` and `vect_memmap`
    depending on whether the array is stored in memory or on disk.  The raw
    `vect` class is not a valid class by itself.

    The vector representation of the array is the flattend array.

    One of the features that `vect`s implement is named axes.  This allows
    maps to carry axis information with them, among other things.  For
    `vect`s axis names must be unique (This is not true of `mat`s).

    Parameters
    ----------
    input_array: InfoArray (for vect_array) or InfoMemmap (for vect_memmap)
        Array to be converted to a vect.
    axis_names: tuple of strings, optional
        The sequence contains the name of each axis.  This sequence will be
        stored in the `axes` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the axis names.

    Attributes
    ----------
    axes: tuple of strings
        The names of each of the axes of the array.

    See Also
    --------
    make_vect: Cast any array as a vector.
    InfoArray, info_memap: Base classes that handle meta data.

    Notes
    -----
    Since much of the functionality provided by this class is only valid
    for a certain shape of the array, shape changing operations in
    general return an `InfoArray` or `InfoMemmap` as appropriate (except
    explicit assignment to vect.shape).

    The `axes` attribute is actually stored in the `InfoArray`'s info
    dictionary.  This is just an implimentation detail.
    """

    __array_priority__ = 2.0

    # self.info_base is set in the class factory.
    def __new__(cls, input_array, axis_names=None):
        if not isinstance(input_array, cls.info_base):
            raise ValueError("Array to convert must be instance of " +
                             str(cls.info_base))

        obj = input_array.view(cls)

        if 'type' in obj.info:
            if axis_names is not None:
                warnings.warn("Initialization argument ignored. Requisite "
                              "metadata for vector already exists. "
                              "Clear info dictionary if you want opposite "
                              "behaviour.")

            if obj.info['type'] != 'vect':
                raise ValueError("Meta data present is incompatible.")

            helpers.check_axis_names(obj)

        else:
            helpers.set_type_axes(obj, 'vect', axis_names)

        return obj

    def __setattr__(self, name, value):
        if name == 'axes':
            helpers.check_axis_names(self, value)
            self.info['axes'] = value
        else:
            self.info_base.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'axes':
            return self.info['axes']
        else:
            # Since numpy uses __get_attribute__ not __getattr__, we should
            # raise an Attribute error.
            raise AttributeError("Attribute " + name + " not found.")

    def flat_view(self):
        """Returns a view of the vector that has been flattened.

        The view will be cast as a scipy.ndarray and its shape will be
        (self.size, ).  It is a view, so writing to it will write back to the
        original vector object.

        Returns
        -------
        flat_view: np.ndarray
            A view of `self` as an ndarray, flattened to 1D.
        """
        flat = self.view(sp.ndarray)
        flat.shape = (self.size, )
        return flat

    def mat_shape(self):
        """Get the shape of the represented matrix (vector)."""
        helpers.check_axis_names(self)
        return (self.size,)


def _vect_class_factory(base_class):
    """Internal class factory for making a vector class that inherits from
    either InfoArray or InfoMemmap."""

    if (base_class is not info_header.InfoArray) and \
       (base_class is not info_header.InfoMemmap):
        raise TypeError("Vectors inherit from info arrays or info memmaps.")

    class vect_class(VectorObject, base_class):
        __doc__ = VectorObject.__doc__
        info_base = base_class

    return vect_class


vect_array = _vect_class_factory(info_header.InfoArray)
vect_array.__name__ = 'vect_array'
vect_memmap = _vect_class_factory(info_header.InfoMemmap)
vect_memmap.__name__ = 'vect_memmap'


def make_vect(array, axis_names=None):
    """Do whatever it takes to make a vect out of an array.

    Convert any class that can be converted to a vect (array, InfoArray,
    memmap, InfoMemmap) to the appropriate vect object (vect_array,
    vect_memmap).

    This convenience function just simplifies the constructor hierarchy.
    Normally to get a vect out of an array, you would need to construct an
    intermediate InfoArray object.  This bypasses that step.

    Parameters
    ----------
    array: array_like
        Array to be converted to vect object (if possible).
    axis_names: tuple of strings, optional
        The sequence contains the name of each axis.  This sequence will be
        stored in the `axes` attribute.  This parameter is ignored if
        `input_array`'s info attribute already contains the axis names.

    Returns
    -------
    vect_arr: vect_array or vect_memmap
        A view of `array` converted to a vect object.

    """

    if isinstance(array, sp.memmap):
        if not isinstance(array, info_header.InfoMemmap):
            array = info_header.InfoMemmap(array)
        return vect_memmap(array, axis_names)
    elif isinstance(array, sp.ndarray):
        if not isinstance(array, info_header.InfoArray):
            array = info_header.InfoArray(array)
        return vect_array(array, axis_names)
    else:
        raise TypeError("Object cannot be converted to a vector.")
