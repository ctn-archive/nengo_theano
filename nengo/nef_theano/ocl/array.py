import numpy as np
import pyopencl as cl

class Array(object):
    def __init__(self, data, dtype, shape, strides):
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        assert type(self.shape) == list
        assert type(self.strides) == list

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def itemstrides(self):
        rval = []
        for si in self.strides:
            assert 0 == si % self.dtype.itemsize
            rval.append(si // self.dtype.itemsize)
        return rval

    def empty_like(self):
        buf = cl.Buffer(self.data.context,
            flags=cl.mem_flags.READ_WRITE,
            size=self.data.size)
        return Array(buf,
                dtype=self.dtype,
                shape=list(self.shape),
                strides=list(self.strides))

    def __str__(self):
        return '%s{%s, shape=%s}' % (
                self.__class__.__name__, self.dtype, self.shape)

    def __repr__(self):
        return '%s{%s, shape=%s, strides=%s}' % (
                self.__class__.__name__, self.dtype, self.shape, self.strides)


def ocldtype(obj):
    if isinstance(obj, basestring):
        return {
            'float32': 'float',
            'float64': 'double',
            'int64': 'long',
        }[obj]
    else:
        raise NotImplementedError('ocldtype', obj)


def to_device(queue, arr, flags=cl.mem_flags.READ_WRITE):
    arr = np.asarray(arr)
    buf = cl.Buffer(queue.context, flags, size=len(arr.data))
    cl.enqueue_copy(queue, buf, arr.data).wait()
    return Array(buf, arr.dtype, list(arr.shape), list(arr.strides))


def empty(context, shape, dtype, flags=cl.mem_flags.READ_WRITE,
        strides=None, order='C'):

    dtype = np.dtype(dtype)
    
    if strides is None:
        strides = [dtype.itemsize]
        for shp in shape:
            strides.append(strides[-1] * shp)
        size = strides[-1]
        strides = strides[:-1]
        if order == 'C':
            strides.reverse()
    buf = cl.Buffer(context, flags, size=size)
    return Array(buf, dtype, list(shape), list(strides))


