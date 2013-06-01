import vector
import scipy as sp

tvec = vector.make_vect(sp.zeros((5, 5)), axis_names=('x', 'y'))
tvec.set_axis_info('x', 2, 0.5)
print tvec.get_axis('x')
#xspec_arr = vector.make_vect(xspec, axis_names=k_axes)
#xspec_arr.info = info
