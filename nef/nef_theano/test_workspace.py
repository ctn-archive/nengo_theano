import numpy as np
from theano import tensor
from workspace import Workspace

def test_basic_1():


    x = tensor.vector('x')
    y = tensor.vector('y')

    ws = Workspace()
    ws[x] = [1, 2]
    ws[y] = [3, 4]
    ws.compile_update('f', [
        (x, 2 * x),
        (y, x + y)])
    
    assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[2, 4], [4, 6]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[4, 8], [6, 10]])

