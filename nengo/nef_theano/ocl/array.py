import numpy as np
import pyopencl as cl

class Array(object):
    def __init__(self, data, dtype, shape, strides, offset=0):
        self.data = data
        self.dtype = np.dtype(dtype)
        self.shape = map(int, shape) # -- makes new list too
        self.strides = map(int, strides) # -- makes new list too
        self.offset = offset
        assert type(self.shape) == list

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
        return '%s{%s, shape=%s, strides=%s, bufsize=%i, offset=%s}' % (
            self.__class__.__name__, self.dtype, self.shape, self.strides,
            self.data.size, self.offset)

    @property
    def ocldtype(self):
        return ocldtype(self.dtype)


def ocldtype(obj):
    obj = str(obj)
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
        strides = [int(dtype.itemsize)]
        if order.upper() == 'C':
            for shp in reversed(shape):
                strides = [strides[0] * int(shp)] + strides
            bufsize = strides[0]
            strides = strides[1:]
        elif order.upper() == 'F':
            raise NotImplementedError()
        else:
            raise ValueError(order)
    try:
        buf = cl.Buffer(context, flags, size=bufsize)
    except:
        print context, flags, type(flags), bufsize, type(bufsize)
        raise
    return Array(buf, dtype, list(shape), list(strides))


