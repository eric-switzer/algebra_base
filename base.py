import scipy as sp
import cubic_conv_interpolation as cci


class AlgObject(object):
    """Base class for all vectors and matricies.

    This is not an actual class by itself, just defines some methods common to
    both `mat` objects and `vect` objects.
    """

    def __array_finalize__(self, obj):
        self.info_base.__array_finalize__(self, obj)
        if (obj is not None) and (self.shape != obj.shape):
            self.__class__ = self.info_base

    def set_axis_info(self, axis_name, centre, delta):
        """Set meta data for calculating values of an axis.

        This provides the meta data required to calculate one of the axes.
        This data is stored in the `info` attribute, which is carried
        around by this class and easily written to disk.

        The information provided is subsequently used in `get_axis` to
        calculate the values along a given axis.

        Parameters
        ----------
        axis_name: str
            Name of the axis for which you are setting the meta data.
            Must match one of the entries of the axes attribute.
        centre: float
            The value of the axis at the centre bin (indexed by n//2),
            where n = self.shape[i] and self.axes[i] is `axis_name`.
        delta: float
            The width of each bin.

        See Also
        --------
        copy_axis_info
        get_axis

        Examples
        --------
        >>> import vector
        >>> a = vector.make_vect(sp.zeros(5, 5), axis_names=('ra', 'dec'))
        >>> a.set_axis_info('ra', 2, 0.5)
        >>> a.get_axis('ra')
        array([1.0, 1.5, 2.0, 2.5, 3.0])
        """

        if axis_name not in self.axes:
            raise ValueError("axis_name not in self.axes.")

        self.info[axis_name + '_centre'] = float(centre)
        self.info[axis_name + '_delta'] = float(delta)

    def copy_axis_info(self, alg_obj):
        """Set the axis info by copying from another AlgObject.

        This transfers meta data that is set with `set_axis_info` another
        AlgObject instance.

        Parameters
        ----------
        alg_obj: base.AlgObject instance
            Object from which to copy axis meta data.  Meta data for all axis
            names that occur in both `alg_obj.axes` and `self.axes` is copied.

        See Also
        --------
        set_axis_info
        get_axis
        """

        if isinstance(alg_obj, dict):
            info = alg_obj
        else:
            info = alg_obj.info

        for axis in self.axes:
            if axis in info["axes"]:
                try:
                    centre = info[axis + '_centre']
                    delta = info[axis + '_delta']
                except KeyError:
                    continue
                self.info[axis + '_centre'] = centre
                self.info[axis + '_delta'] = delta

    def get_axis(self, axis_name):
        """Calculate the array representing a named axis.

        For a given axis name, calculate the 1D array that gives the value of
        that axis.  This requires that the relevant meta data be set by
        `set_axis_info`.

        Parameters
        ----------
        axis_name: str or int
            Name of the axis to be calculated.  `axis_name` must occur in the
            `axes` attribute.  If an int is passed, than it is convered to a
            string by indexing the `axes` attribute.

        Returns
        -------
        axis_array: np.ndarray
            The array corresponding to the values of the axis quantity along
            it's axis.

        See Also
        --------
        set_axis_info
        copy_axis_info
        """

        if isinstance(axis_name, int):
            axis_name = self.axes[axis_name]

        len = self.shape[self.axes.index(axis_name)]

        return (self.info[axis_name + '_delta']*(sp.arange(len) - len//2)
                + self.info[axis_name + '_centre'])

    def slice_interpolate_weights(self, axes, coord, kind='linear'):
        """Get the interpolation weights for a subset of the dimensions.

        This method gets the interpolation weights for interpolating the
        AlgObject is some subset of its dimensions.  This provides the
        freedom in the uninterpolated dimensions to either slice or otherwise
        index the array.

        Note: When using the cubic convolution interpolation, the points
        may out of bounds. That is how the algorithm works when interpolating
        right beside a boundary. If the value of those nodes are needed, use
        'get_value' from the cci helper module.
        Also, some weights will be negative as needed by the algorithm.

        Parameters
        ----------
        axes: int or sequence of ints (length N)
            Over which axes to interpolate.
        coord: float or sequence of floats (length N)
            The coordinate location to interpolate at.
        kind: string
            The interpolation algorithm.  Options are: 'linear' or 'nearest'.
        And now 'cubic' too!

        Returns
        -------
        points: array of ints shape (M x N)
            The indices for the N `axes` at the M interpolation data points
            that are used.
        weights: array of floats length M
            Weights for the interpolations data points.
        """

        if not hasattr(axes, '__iter__'):
            axes = (axes,)

        if not hasattr(coord, '__iter__'):
            coord = (coord,)

        n = len(axes)

        if n != len(coord):
            message = "axes and coord parameters must be same length."
            raise ValueError(message)

        if kind in ('linear',):
            # Any interpolation scheme that only depends on data points
            # directly surrounding. There are 2^n of them.
            m = 2**n
            points = sp.empty((m, n), dtype=int)
            weights = sp.empty((m,), dtype=float)
            # Find the indices of the surrounding points, as well as the
            single_inds = sp.empty((n, 2), dtype=int)
            normalized_distance = sp.empty((n, 2), dtype=float)

            for ii in range(n):
                axis_ind = axes[ii]
                value = coord[ii]
                # The spacing between points of the axis we are considering.
                delta = abs(self.info[self.axes[axis_ind] + "_delta"])
                # For each axis, find the indicies that surround the
                # interpolation location.
                axis = self.get_axis(axis_ind)
                if value > max(axis) or value < min(axis):
                    message = ("Interpolation coordinate outside of "
                               "interpolation range.  axis: " + str(axis_ind)
                               + ", coord: " + str(value) + ", range: "
                               + str((min(axis), max(axis))))

                    raise ce.DataError(message)

                distances = abs(axis - value)
                min_ind = distances.argmin()
                single_inds[ii, 0] = min_ind
                normalized_distance[ii, 0] = distances[min_ind]/delta
                distances[min_ind] = distances.max() + 1
                min_ind = distances.argmin()
                single_inds[ii, 1] = min_ind
                normalized_distance[ii, 1] = distances[min_ind]/delta

            # Now that we have all the distances, figure out all the weights.
            for ii in range(m):
                temp_ii = ii
                weight = 1.0

                for jj in range(n):
                    points[ii, jj] = single_inds[jj, temp_ii % 2]
                    weight *= (1.0 - normalized_distance[jj, temp_ii % 2])
                    temp_ii = temp_ii//2

                weights[ii] = weight

        elif kind == 'nearest':
            # Only one grid point to consider and each axis is independant.
            m = 1
            weights = sp.ones(1, dtype=int)
            points = sp.empty((1, n), dtype=int)

            # Loop over the axes we are interpolating.
            for ii in range(n):
                axis_name = self.axes[axes[ii]]
                axis_centre = self.info[axis_name + "_centre"]
                axis_delta = self.info[axis_name + "_delta"]
                index = (coord[ii] - axis_centre)/axis_delta
                index += self.shape[axes[ii]]//2
                if index < 0 or index > self.shape[axes[ii]] - 1:
                    message = ("Interpolation coordinate outside of "
                               "interpolation range.  axis: " + str(axes[ii])
                               + ", coord: " + str(coord[ii]) + ".")

                    raise ce.DataError(message)

                points[0, ii] = round(index)

        elif kind == 'cubic':
            # Make sure the given point is an array.
            pnt = sp.array(coord)
            # Get the array containing the first value in each dimension.
            # And get the arrays of deltas for each dimension.
            x0 = []
            step_sizes = []

            for ax in axes:
                ax_name = self.axes[ax]
                x0.append(self.get_axis(ax_name)[0])
                ax_delta_name = ax_name + "_delta"
                step_sizes.append(self.info[ax_delta_name])

            x0 = sp.array(x0)
            step_sizes = sp.array(step_sizes)
            # Get the maximum possible index in each dimension in axes.
            max_inds = np.array(self.shape) - 1
            max_needed_inds = []

            for ax in axes:
                max_needed_inds.append(max_inds[ax])

            max_inds = np.array(max_needed_inds)
            # If there are less than four elements along some dimension
            # then raise an error since cubic conv won't work.
            too_small_axes = []
            for i in range(len(max_inds)):
                if max_inds[i] < 3:
                    too_small_axes.append(axes[i])

            if len(too_small_axes) != 0:
                msg = "Need at least 4 points for cubic interpolation " + \
                      "on axis (axes): " + str(too_small_axes)
                raise ce.DataError(msg)

            # Get the nodes needed and their weights.
            points, weights = cci.interpolate_weights(axes, pnt, x0,
                                                      step_sizes, max_inds)
        else:
            message = "Unsupported interpolation algorithm: " + kind
            raise ValueError(message)

        return points, weights

    def slice_interpolate(self, axes, coord, kind='linear'):
        """Interpolate along a subset of dimensions.

        This method interpolate the array object along some subset of it's
        dimensions.  The array is sliced long the uninterpolated dimensions.

        Parameters
        ----------
        axes: int or sequency of ints (length N)
            Over which axes to interpolate.
        coord: float or sequence of floats (length N)
            The coordinate location to interpolate at.
        kind: string
            The interpolation algorithm.  Options are: 'linear', 'nearest'.

        Returns
        -------
        slice: numpy array
            The array has a shape corresponding to the uninterpolated
            dimensions of `self`.  That is it has dimensions the same as `self`
            along dimension included in `axes` which are interpolated.
        """

        if not hasattr(axes, '__iter__'):
            axes = (axes,)

        if not hasattr(coord, '__iter__'):
            coord = (coord,)

        n = len(axes)

        if n != len(coord):
            message = "axes and coord parameters must be same length."
            raise ValueError(message)

        # Get the contributing points and their weights.
        points, weights = self.slice_interpolate_weights(axes, coord, kind)

        # Sum up the points
        q = points.shape[0]
        out = 0.0

        for ii in range(q):
            index = [slice(None)] * self.ndim
            for jj, axis in enumerate(axes):
                index[axis] = points[ii, jj]

            index = tuple(index)
            out += weights[ii] * self[index]

        return out


if __name__ == "__main__":
    import doctest

    OPTIONFLAGS = (doctest.ELLIPSIS |
                   doctest.NORMALIZE_WHITESPACE)
    #doctest.testmod(optionflags=OPTIONFLAGS)
