import copy
import sys

import numpy as np

import theano
from theano.gof.vm import VM_Linker
from theano.compile import deep_copy_op
from theano.compile.pfunc import rebuild_collect_shared


def optimizer_from_any(specifier):
    if isinstance(specifier, basestring):
        try:
            dct = theano.compile.mode.predefined_optimizers
            query = dct[specifier]
        except KeyError:
            raise ValueError('Optimizer %s not in %s' % (
                specifier, dct))
        return theano.compile.mode.optdb.query(query)
    else:
        # XXX probably not implemented error is more appropriate
        raise TypeError(specifier)


class UpdateFGraph(object):
    def __init__(self, updated_vars, givens=None):
        """
        updated_vars: sequence of (dst, expr) pairs
        vals_memo: dict Variable -> [value]

        """

        # -- unique_outputs is used here to ensure that there is some
        #    double-buffering going on, because actually dests and outputs can
        #    include some of the same variables (e.g. swap values)
        dests, outputs = zip(*updated_vars)
        unique_outputs = map(deep_copy_op, outputs)

        # -- partial graph clone to use givens
        stuff = rebuild_collect_shared(
            unique_outputs,
            inputs=[],
            replace=givens,
            rebuild_strict=True,
            copy_inputs_over=True)
        _inputs, unique_outputs_w_givens, other_stuff = stuff
        clone_equiv1, _update_d, _update_expr, _shared_inputs = other_stuff

        all_inputs = theano.gof.graph.inputs(unique_outputs_w_givens)

        # -- full graph clone to protect original graph
        clone_equiv = {}
        theano.gof.graph.clone_get_equiv(
            all_inputs,
            unique_outputs_w_givens,
            copy_inputs_and_orphans=True,
            memo=clone_equiv)
        for orig_var in clone_equiv1:
            clone_equiv[orig_var] = clone_equiv[clone_equiv1[orig_var]]
        self.cloned_inputs = [clone_equiv[var] for var in all_inputs]
        self.cloned_dests = [clone_equiv[var] for var in dests]
        self.cloned_outputs = [clone_equiv[var] for var in unique_outputs_w_givens]
        fgraph = theano.gof.fg.FunctionGraph(
            self.cloned_inputs,
            self.cloned_outputs)

        # -- load up fgraph with features necessary to maintain correctness:
        for node in fgraph.apply_nodes:
            if getattr(node.op, 'destroy_map', None):
                if not accept_inplace:
                    raise TypeError("Graph must not contain inplace operations", node, node.op)
                else:
                    fgraph.attach_feature(theano.gof.DestroyHandler())
                    break

        # We need to protect all immutable inputs from inplace operations.
        fgraph.attach_feature(
                theano.compile.function_module.Supervisor(invar
                    for invar in self.cloned_inputs
                    if not ((invar in self.cloned_dests) or
                            (hasattr(fgraph, 'destroyers') and
                                fgraph.destroyers(input)))))

        # If named nodes are replaced, keep the name
        for feature in theano.compile.function_module.std_fgraph.features:
            fgraph.attach_feature(feature())

        self.updated_vars = updated_vars
        self.all_inputs = all_inputs
        self.outputs = outputs
        self.unique_outputs = unique_outputs
        self.clone_equiv = clone_equiv
        self.fgraph = fgraph


class CompiledUpdate(object):
    def __init__(self, ufgraph, vals_memo, profiler=None, **VM_Linker_kwargs):

        # -- create a VM to run the updates
        #    XXX CVM is necessary here until LoopGC implements updates
        linker = VM_Linker(use_cloop=True, **VM_Linker_kwargs)
        linker.accept(ufgraph.fgraph, no_recycling=[])
        linker.accept_var_updates(dict(zip(
            ufgraph.cloned_dests,
            ufgraph.cloned_outputs)))

        input_storage = [vals_memo[i] if i in vals_memo else [i.data]
                for i in ufgraph.all_inputs]

        vm, input_containers, output_containers, thunks, order = linker.make_all(
            profiler=profiler,
            input_storage=input_storage,
            )

        self.ufgraph = ufgraph
        self.vals_memo = vals_memo
        self.input_storage = input_storage
        self.vm = vm
        self.input_containers = input_containers
        self.output_containers = output_containers
        self.thunks = thunks
        self.order = order

    def __call__(self):
        # if profiler then we need to update it (see function_module.py:641)
        return self.vm()


class Workspace(object):
    """
    """

    def __init__(self):
        self.vals_memo = {}
        self.compiled_updates = {}

    def __contains__(self, key):
        return key in self.vals_memo

    def __getitem__(self, key):
        return self.vals_memo[key][0]

    def __setitem__(self, key, val):
        filtered_val = key.type.filter(val, strict=False, allow_downcast=True)
        if key in self.vals_memo:
            self.vals_memo[key][0] = filtered_val
        else:
            self.vals_memo[key] = [filtered_val]

    def compile_update(self, key, updated_vars, optimizer=None):
        """

        Return a function that will update this workspaces
        values for keys according to `exprs`

        """
        ufgraph = UpdateFGraph(updated_vars)
        if optimizer:
            ufgraph.optimize(optimizer)
        cu = CompiledUpdate(ufgraph, self.vals_memo)
        self.compiled_updates[key] = cu
        return cu

    def run_update(self, key):
        self.compiled_updates[key]()

    def optimize(self, specifier):
        optimizer = optimizer_from_any(specifier)
        for key, cu in self.compiled_updates.items():
            optimizer.apply(cu.ufgraph.fgraph)
            cu_opt = CompiledUpdate(cu.ufgraph, self.vals_memo)
            self.compiled_updates[key] = cu_opt


class SharedStorageWorkspace(Workspace):
    def __init__(self, ws):
        Workspace.__init__(self)

        self.views_memo = {}

        # set up some views
        self.s_fvector = theano.tensor.vector('fvector')
        vectors = [var for var in ws.vals_memo
                if var.type == self.s_fvector.type]
        if vectors:
            fvector = np.concatenate(
                    [ws.vals_memo[var][0] for var in vectors]).astype('float32')
            offset = 0
            for var in vectors:
                self.views_memo[var] = (
                        self.s_fvector,
                        offset,
                        len(ws.vals_memo[var][0]))
                offset += len(ws.vals_memo[var][0])
            self.vals_memo[self.s_fvector] = [fvector]

        #print self.views_memo

        # set up some normal values
        for var in ws.vals_memo:
            if var not in self.views_memo:
                self.vals_memo[var] = copy.deepcopy(ws.vals_memo[var])

        for fname, f in ws.compiled_updates.items():
            self.compile_update(fname, f.ufgraph.updated_vars)

    def __contains__(self, key):
        return key in self.vals_memo or key in self.views_memo

    def __getitem__(self, key):
        if key in self.views_memo:
            var, offset, n = self.views_memo[key]
            return self[var][offset: offset + n]
        else:
            return self.vals_memo[key][0]

    def __setitem__(self, key, val):
        filtered_val = key.type.filter(val, strict=False, allow_downcast=True)

        if key in self.views_memo:
            var, offset, n = self.views_memo[key]
            self.vals_memo[var][0][offset: offset + n] = filtered_val
        else:
            if key in self.vals_memo:
                self.vals_memo[key][0] = filtered_val
            else:
                self.vals_memo[key] = [filtered_val]

    def compile_update(self, key, updated_vars):
        new_s_fvector = self.s_fvector
        no_view_updated_vars = []
        for dst, out in updated_vars:
            if dst in self.views_memo:
                var, offset, n_elems = self.views_memo[dst]
                assert var is self.s_fvector # XXX HACK
                new_s_fvector = theano.tensor.set_subtensor(
                        new_s_fvector[offset: offset + n_elems],
                        out)
            else:
                no_view_updated_vars.append((dst, out))

        if new_s_fvector != self.s_fvector:
            no_view_updated_vars.append((self.s_fvector,
                new_s_fvector))

        givens = []
        for var in self.views_memo:
            svar, offset, n_elems = self.views_memo[var]
            givens.append((var, svar[offset: offset + n_elems]))

        ufgraph = UpdateFGraph(no_view_updated_vars, givens=givens)
        cu = CompiledUpdate(ufgraph, self.vals_memo)
        #cu = CompiledUpdate(no_view_updated_vars, self.vals_memo,
                #givens=givens)
        self.compiled_updates[key] = cu
        return cu


class Function(Workspace):
    """
    Special case of Workspace for implementing a single callable expression

    TODO: Provides support for structuring outputs as nested list, dict, etc.
    """
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        self._dests = [o.type() for o in outputs]
        for var in self._inputs + _self.dests:
            self[var] = None
        self.add_compiled_update('__call__', zip(self._dests, self._outputs))

    def __call__(self, *args):
        assert len(self._inputs) == len(args)
        for var, val in zip(self._inputs, args):
            self[var] = val
        self.compiled_updates['__call__']()
        return [self[var] for var in self._dests]


