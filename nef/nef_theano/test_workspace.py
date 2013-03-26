import unittest
import numpy as np
import theano
from theano import tensor
from workspace import Workspace, SharedStorageWorkspace

class StdMixins(object):
    def test_scaffolding(self):
        pass

    def test_optimize(self):
        ws = self.foo[2]
        ws.optimize('fast_run')


class SimpleGraph(unittest.TestCase, StdMixins):
    def setUp(self):
        x = tensor.vector('x')
        y = tensor.vector('y')

        ws = Workspace()
        ws[x] = [1, 2]
        ws[y] = [3, 4]
        ws.compile_update('f', [
            (x, 2 * x),
            (y, x + y)])
        self.foo = x, y, ws

    def tearDown(self):
        x, y, ws = self.foo

        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])
        ws.run_update('f')
        assert np.allclose([ws[x], ws[y]],[[2, 4], [4, 6]])
        ws.run_update('f')
        assert np.allclose([ws[x], ws[y]],[[4, 8], [6, 10]])


class SwapGraph(unittest.TestCase, StdMixins):
    def setUp(self):
        x = tensor.vector('x')
        y = tensor.vector('y')

        ws = Workspace()
        ws[x] = [1, 2]
        ws[y] = [3, 4]
        ws.compile_update('f', [(x, y), (y, x)])
        self.foo = x, y, ws

    def tearDown(self):
        x, y, ws = self.foo
        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])
        ws.run_update('f')
        assert np.allclose([ws[x], ws[y]],[[3, 4], [1, 2]])
        ws.run_update('f')
        assert np.allclose([ws[x], ws[y]],[[1, 2], [3, 4]])


class MergeableGraph(unittest.TestCase, StdMixins):
    def setUp(self):

        x = tensor.vector('x')
        y = tensor.vector('y')

        ws = Workspace()
        ws[x] = [1, 2]
        ws[y] = [3, 4]
        f = ws.compile_update('f', [(x, 2 * x), (y, 2 * y)])

        ws_shrd = SharedStorageWorkspace(ws)
        f_opt = ws_shrd.compiled_updates['f']
        self.foo = x, y, ws, ws_shrd

    def tearDown(self):
        x, y, ws, ws_shrd = self.foo

        for w in (ws, ws_shrd):
            assert np.allclose([w[x], w[y]],[[1, 2], [3, 4]])
            w.run_update('f')
            assert np.allclose([w[x], w[y]],[[2, 4], [6, 8]])
            w.run_update('f')
            assert np.allclose([w[x], w[y]],[[4, 8], [12, 16]])

    def test_merged(self):
        ws, ws_shrd = self.foo[2:]
        assert len(ws.vals_memo) == 2
        assert len(ws_shrd.vals_memo) == 1

    def test_computation_merged(self):

        ws_shrd = self.foo[3]
        ws_shrd.optimize('fast_run')
        theano.printing.debugprint(ws_shrd.compiled_updates['f'].ufgraph.fgraph.outputs)
