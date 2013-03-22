import sys

import theano
from theano.gof.vm import VM_Linker

class CompiledUpdate(object):
    def __init__(self, updated_vars, vals_memo, profiler=None,
              **VM_Linker_kwargs):
        """
        updated_vars: sequence of (dst, expr) pairs

        allow_gc - force the virtual machine to clean up unnecessary references,
            in order to allow garbage collection on intermediate values during
            computation of a function.

        use_cloop - use the C-based virtual machine if possible

        callback - a callable object to call after each call to a thunk within
            the virtual machine.  It will be called with four arguments called
            'node', 'thunk', 'storage_map', and 'compute_map'.

        This object does *NOT* clone the expression graph, nor does it modify
        the expression graph.

        # XXX currently it does modify the expression graph because it creates an FGraph
        # instance, but it should be changed to avoid this, because all it really needs
        # the fgraph for is sorting.

        """

        dests, outputs = zip(*updated_vars)
        inputs = theano.gof.graph.inputs(outputs + dests)
        fgraph = theano.gof.FunctionGraph(inputs, outputs)

        #print 'inputs'
        #theano.printing.debugprint(inputs)
        #print 'outputs'
        #theano.printing.debugprint(outputs)

        linker = VM_Linker(**VM_Linker_kwargs)
        linker.accept(fgraph, no_recycling=[])
        linker.accept_var_updates(dict(updated_vars))

        input_storage = [vals_memo[i] if i in vals_memo else [i.data]
                for i in inputs]
        output_storage = [vals_memo[i] if i in vals_memo else [None]
                for i in dests] # CAREFUL IS THIS OK?

        vm, input_containers, output_containers, thunks, order = linker.make_all(
            profiler=profiler,
            input_storage=input_storage,
            output_storage=output_storage)

        self.inputs = inputs
        self.outputs = outputs
        self.fgraph = fgraph
        self.vm = vm
        self.input_storage = input_storage
        self.output_storage = output_storage
        self.input_containers = input_containers
        self.output_containers = output_containers
        self.thynks = thunks
        self.order = order

    def __call__(self):
        # if profiler then we need to update it (see function_module.py:641)
        return self.vm()


class Workspace(object):
    """
    """

    def __init__(self ):
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

    def compile_update(self, key, updated_vars):
        """

        Return a function that will update this workspaces
        values for keys according to `exprs`

        """
        cu = CompiledUpdate(updated_vars, self.vals_memo)
        self.compiled_updates[key] = cu
        return cu

    def run_update(self, key):
        self.compiled_updates[key]()

    def optimize_memory_layout(self):
        """
        Potentially optimize the internal physical memory layout of variables
        in order to run the compiled_updates faster.

        This function generally invalidates any views into memory associated
        with variables.
        """
        pass


class Function(Workspace):
    """
    Special case of Workspace for implementing a single callable expression
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


