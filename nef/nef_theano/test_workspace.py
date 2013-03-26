import numpy as np
import theano
from theano import tensor
from workspace import Workspace, SharedStorageWorkspace

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


def test_no_alias():
    x = tensor.vector('x')
    y = tensor.vector('y')

    ws = Workspace()
    ws[x] = [1, 2]
    ws[y] = [3, 4]
    ws.compile_update('f', [(x, y), (y, x)])

    assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[3, 4], [1, 2]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])


def test_merge_1():
    x = tensor.vector('x')
    y = tensor.vector('y')

    ws = Workspace()
    ws[x] = [1, 2]
    ws[y] = [3, 4]
    f = ws.compile_update('f', [(x, 2 * x), (y, 2 * y)])

    ws_opt = SharedStorageWorkspace(ws)
    f_opt = ws_opt.compiled_updates['f']

    print f_opt.order

    assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[2, 4], [6, 8]])
    ws.run_update('f')
    assert np.allclose([ws[x], ws[y]],[[4, 8], [12, 16]])

    assert np.allclose([ws_opt[x], ws_opt[y]],[[1, 2], [3, 4]])
    ws_opt.run_update('f')
    assert np.allclose([ws_opt[x], ws_opt[y]],[[2, 4], [6, 8]])
    ws_opt.run_update('f')
    assert np.allclose([ws_opt[x], ws_opt[y]],[[4, 8], [12, 16]])

