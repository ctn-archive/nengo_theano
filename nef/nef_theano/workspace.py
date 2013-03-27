import copy
import sys

import numpy as np

import theano
from theano.gof.vm import VM_Linker
from theano.compile import deep_copy_op
from theano.compile.function_module import infer_reuse_pattern
from theano.compile.pfunc import rebuild_collect_shared
from theano.printing import debugprint


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
        # TODO probably not implemented error is more appropriate
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

        fgraph.attach_feature(theano.tensor.opt.ShapeFeature())

        # -- pre-install the shape information from the Hints created by
        #    e.g. SharedStorageWorkspace
        done = {}
        for node in fgraph.toposort():
            if is_hint_node(node):
                if node.inputs[0] in done: continue
                hints = dict(node.op.hints)
                if 'shape' in hints:
                    x = node.inputs[0]
                    assert x.ndim == len(hints['shape'])
                    if x in done:
                        assert done[x] == hints['shape']
                    else:
                        var_shape = tuple(
                            map(theano.tensor.as_tensor_variable,
                                hints['shape']))
                        fgraph.shape_feature.shape_of[node.inputs[0]] = var_shape
                        done[x] = hints['shape']

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
        linker = VM_Linker(use_cloop=False, allow_gc=False, **VM_Linker_kwargs)
        no_recycling = infer_reuse_pattern(ufgraph.fgraph, ufgraph.fgraph.outputs)
        linker.accept(ufgraph.fgraph, no_recycling=no_recycling)
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

    This workspace is meant to be serializable, at least before it has been
    optimized.

    Recommended workflow for many repeated evaluations (pre-profile-driven
    optimization engine):
    1. build this type of workspace to define a function system
    2. use it to initialize a SharedStorageWorkspace, which will optimize the
       memory layout.
    3. call ws.optimize() on the SharedStorageWorkspace to optimize the
       computational graph for the optimized physical layout.
    4. run the optimized function system many times, it is the fastest.
    5. when it comes time to save, call ws.update(fast_ws) to bring the values
       back from the fast workspace to the original (slow) one, and save the
       slow one.
    """

    def __init__(self):
        self.vals_memo = {}
        self.compiled_updates = {}

    def __iter__(self):
        return self.vals_memo.keys()

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

    def update(self, other):
        for key in other:
            self[key] = other[key]

    # XXX unfortunate name!
    # -- since Workspace is like a value dictionary, "update" is defined by
    #    Python convention to mean a dictionary update.
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

from theano.sandbox.linalg.ops import Hint
from theano.sandbox.linalg.ops import is_hint_node

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
        noview_updated_vars = dict() #XXX want ordered-dict here
        for dst, out in updated_vars:
            if dst in self.views_memo:
                var, offset, n_elems = self.views_memo[dst]
                # -- build the shape into the graph
                #shp = self.vals_memo[var][0].shape
                # print 'shp', shp
                upvar = noview_updated_vars.get(var, var)
                upvar = theano.tensor.set_subtensor(
                        upvar[offset: offset + n_elems],
                        out)
                noview_updated_vars[var] = upvar
            else:
                if dst in noview_updated_vars:
                    raise ValueError('duplicate destination', updated_vals)
                noview_updated_vars[dst] = out

        givens = []
        for var in self.views_memo:
            svar, offset, n_elems = self.views_memo[var]
            shp = self.vals_memo[svar][0].shape
            svar = Hint(shape=shp)(svar)
            givens.append((var, svar[offset: offset + n_elems]))

        ufgraph = UpdateFGraph(noview_updated_vars.items(), givens=givens)
        cu = CompiledUpdate(ufgraph, self.vals_memo)
        self.compiled_updates[key] = cu
        return cu


class Function(object):
    """
    Special case of Workspace for implementing a single callable expression

    TODO: Provides support for structuring outputs as nested list, dict, etc.
    """
    # XXX COMPLETELY UNTESTED
    def __init__(self, ws, inputs, outputs, dests, fn_name):
        self._ws = ws
        self._inputs = inputs
        self._outputs = outputs
        self._dests = dests
        self._fn_name = fn_name

    def __call__(self, *args):
        assert len(self._inputs) == len(args)
        for var, val in zip(self._inputs, args):
            self._ws[var] = val
        self._ws.compiled_updates[self._fn_name]()
        # TODO: unflatten dictionaries, singles, nested stuff, etc.
        return [self[var] for var in self._dests]


def function(inputs, outputs, ws_cls=Workspace):
    # XXX COMPLETELY UNTESTED
    ws = ws_cls()
    dests = [o.type() for o in outputs]
    for var in inputs + dests:
        ws[var] = None
    ws.add_compiled_update('__call__', zip(dests, outputs))
    return Function(ws, inputs, outputs, dests, '__call__')


from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_specialize
from theano.tensor.blas import local_optimizer
from theano.tensor.blas import Optimizer

IncSubtensor = theano.tensor.IncSubtensor
Subtensor = theano.tensor.Subtensor
Reshape = theano.tensor.Reshape
from theano.tensor.basic import get_scalar_constant_value
from theano.tensor.basic import NotScalarConstantError


class RefactorSubtensors(Optimizer):
    """
    op(x[a:b]), op(x[b:c]) -> A = op(x[a:c]), A[0:b-a], A[b-a:c-a]

    When some elementwise operation is applied separately to neighbouring
    parts of a tensor, this optimization rearranges things so that the
    elementwise operation is only applied once, and the result is split.
    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(theano.gof.toolbox.ReplaceValidate())

    def apply(self, fgraph):
        nb_iter = 0
        nb_replacement = 0
        nb_replacement_didn_t_remove = 0
        nb_inconsistency_make = 0
        nb_inconsistency_replace = 0
        time_canonicalize = 0
        time_factor_can = 0
        time_factor_list = 0
        time_toposort = 0

        did_something = True
        #print '-- START -- '

        Subtensor = theano.tensor.Subtensor

        # XXX: make sure we don't replace all elemwise(subtensor) with
        # subtensor(elemwise) because most of the time that would be a bad
        # idea!

        while did_something:
            did_something = False
            nb_iter += 1
            #print 'NEW LOOP'

            subtensors = [n for n in fgraph.toposort()
                    if isinstance(n.op, Subtensor)]

            xs_with_subtensor = {}
            for n in subtensors:
                xs_with_subtensor.setdefault(n.inputs[0], []).append(n)

            for x, subtensor_clients in xs_with_subtensor.items():
                if did_something:
                    break
                if len(subtensor_clients) > 1:
                    # -- potentially merge the subtensor clients of x
                    if any(len(n.inputs) > 1 for n in subtensor_clients):
                        # -- TODO: support non-constant indexing ranges
                        continue

                    if all(((len(n.op.idx_list) == 1)
                            and isinstance(n.op.idx_list[0], slice)
                            and isinstance(n.op.idx_list[0].start, int)
                            and isinstance(n.op.idx_list[0].stop, int)
                            and n.op.idx_list[0].step == None
                            )
                            for n in subtensor_clients):
                        ranges = [
                            (n.op.idx_list[0].start, n.op.idx_list[0].stop, n)
                            for n in subtensor_clients]
                        ranges.sort()
                        assert len(ranges) > 1
                        if ranges[0][0] != 0:
                            raise NotImplementedError()
                            # XXX: remember to revise indexing below to be
                            # relative to new vector, so that it will work
                            # when ranges[0].start != 0

                        # -- check if the selection range boundaries match up
                        # TODO: consider merging *some* of the subtensor clients
                        if all(r0[1] == r1[0]
                                for r0, r1 in zip(ranges[:-1], ranges[1:])):
                            #print 'potentially merge', x, ranges
                            replacements = []
                            to_go = set()
                            # -- check for common operations on these slices.
                            # TODO: check for *some* matches
                            for start, stop, subt_node in ranges:
                                for client_apply, pos_in_client in subt_node.outputs[0].clients:
                                    if len(client_apply.outputs) > 1:
                                        raise NotImplementedError()
                                    client_op = client_apply.op
                                    if isinstance(client_op, theano.tensor.Elemwise):
                                        new_inputs = list(client_apply.inputs)
                                        # XXX: need to simultaneously replace
                                        # all new_inputs that our subtensor
                                        # merge is going to affect. If we are
                                        # merging e.g.
                                        #   add(x[1:2], y[1:2])
                                        #   add(x[2:4], y[2:4])
                                        #   -> add(x[1:4], y[1:4])[0:1]
                                        #      add(x[1:4], y[1:4])[1:3]
                                        # then we need to replace both of
                                        # x and y.

                                        new_inputs[pos_in_client] = x
                                        new_out = client_op(*new_inputs)[start:stop]
                                        replacements.append((client_apply.outputs[0], new_out))
                                        assert client_apply.outputs[0] not in to_go
                                        to_go.add(client_apply.outputs[0])
                            #print 'Replacements', replacements
                            fgraph.replace_all_validate(replacements,
                                reason='RefactorSubtensors')
                            nb_replacement += len(replacements)
                            did_something = True
                        else:
                            #print 'clients did not match up'
                            pass
                    else:
                        # -- TODO: match up other kinds of indexing
                        continue

        #theano.printing.debugprint(fgraph.outputs)
        #print '-- DONE -- '
        return (self, nb_iter, nb_replacement, nb_replacement_didn_t_remove,
                nb_inconsistency_make, nb_inconsistency_replace,
                time_canonicalize, time_factor_can,
                time_factor_list, time_toposort)


theano.compile.mode.optdb.register('refactor_subtensors',
        RefactorSubtensors(),
        0, 'fast_compile', 'fast_run')


class Match(object):
    def __init__(self, name, cls,
            match_fn=None,
            unmatch_fn=None,
            **attrs):
        self._name = name
        self._cls = cls
        self._attrs = attrs
        self._match_fn = match_fn
        self._unmatch_fn = unmatch_fn

    def __call__(self, *inputs):
        if hasattr(self, '_inputs'):
            raise TypeError('wrong usage, call Match once')
        self._inputs = inputs
        return self

    def match(self, node, assignment):
        def ok(*args):
            #print 'OK ' + ' '.join([str(a) for a in args])
            pass
        def fail(*args):
            #print '-- ' + ' '.join([str(a) for a in args])
            pass

        if node is None:
            fail('not matching null node')
            return
        if isinstance(node.op, self._cls):
            ok('matched class', node.op, self._cls)
        else:
            fail('not matching wrong class', node.op, self._cls)
            return
        for attr, varname in self._attrs.items():
            opval = getattr(node.op, attr)
            if varname in assignment:
                if assignment[varname] == opval:
                    ok('matched attrib', varname, opval)
                elif (isinstance(opval, (list, tuple)) and
                    isinstance(assignment[varname], (list, tuple))
                    and tuple(opval) == tuple(assignment[varname])):
                    ok('matched attrib', varname, opval)
                else:
                    fail('not matching attribute', varname, opval)
                    return
            else:
                ok('assigning attrib', varname, opval)
                assignment[varname] = opval
        assignment[self._name] = node
        if len(node.inputs) != len(self._inputs):
            raise ValueError('input count')
        for invar, arg in zip(node.inputs, self._inputs):
            if isinstance(arg, Match):
                assignment = arg.match(invar.owner, assignment)
                if not assignment:
                    return
            elif isinstance(arg, MatchConstant):
                assignment = arg.match(invar, assignment)
                if assignment:
                    ok('matched constant', arg, invar)
                else:
                    fail('failed to match constant', arg, invar)
                    return
            elif isinstance(arg, basestring):
                if arg in assignment:
                    if assignment[arg] == invar:
                        ok('matched free var', arg, invar)
                    else:
                        fail('wrong free var', arg, invar)
                        return
                else:
                    ok('assigning free var', arg, invar)
                    assignment[arg] = invar
            else:
                raise NotImplementedError(arg)
        return assignment


class MatchConstant(object):
    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def match(self, var, assignment):
        try:
            value = get_scalar_constant_value(var)
        except NotScalarConstantError:
            return
        if self.val is None:
            # -- any constant value will do
            assignment[self.name] = value
        else:
            if self.val == value:
                assignment[self.name] = value
            else:
                return
        return assignment


@register_canonicalize
@local_optimizer()
def local_consolidate_incsubtensor(node):
    # TODO: check to see if an IncSubtensor among the clients of node would
    # match, and abort if that's true, so we leave the work to the parent.
    # It's not clear if we can get away with not running the entire function
    # fo the parent though... maybe it's better just to write an Optimizer
    # class that iterates correctly and doesn't do too many replacements?
    #
    # -- Empirically, the current local_optimizer seems to be iterating from
    # outputs to inputs, which is actually the order we want.

    template = Match('is1', IncSubtensor, idx_list='i1', set_instead_of_inc='s/i')(
        Match('is2', IncSubtensor, idx_list='i2', set_instead_of_inc='s/i')(
            'x',
            Match('st2', Subtensor, idx_list='i2')('v')),
        Match('st1', Subtensor, idx_list='i1')('v'))

    assignment = template.match(node, {})
    assignments = []
    while assignment:
        assignments.append(assignment)
        assignment = template.match(assignment['is2'], {})

    if not assignments:
        return

    #incsubtensors = [a['is1'] for a in assignments] + [assignments[-1]['is2']]
    if any([len(a['i1']) > 1 for a in assignments]):
        # NotImplemented
        return
    if any([len(a['i2']) > 1 for a in assignments]):
        # NotImplemented
        return
    if any([not isinstance(a['i1'][0], slice) for a in assignments]):
        # NotImplemented
        return
    if any([not isinstance(a['i2'][0], slice) for a in assignments]):
        # NotImplemented
        return
    if any([a['i2'][0].step not in (1, None) for a in assignments]):
        # NotImplementedError
        # It should be possible to detect interleaved access patterns that are
        # guaranteed to provide full coverage (or even just enough coverage to
        # justify doing a more parallized summation)
        # e.g. a[::2] += b[::2] and a[1::2] += b[1::2]
        #      -> (a + b)[::2] and (a + b)[1::2]
        # Like all applications of this optimization, it is meant to really
        # pay off in conjunction with local_cut_whole_incsubtensor, which
        # arises in the graphs formed by SharedStorageWorkspace.
        return

    def start_stop_node(a):
        return (a.op.idx_list[0].start, a.op.idx_list[0].stop, a)

    incsubtensors = ([start_stop_node(a['is1']) for a in assignments]
            + [start_stop_node(assignments[-1]['is2'])])
    incsubtensors.sort()

    # -- ensure we have a contiguous range
    if not all(ssn0[1] == ssn1[0]
            for ssn0, ssn1 in zip(incsubtensors[:-1], incsubtensors[1:])):
        return

    start = incsubtensors[0][0]
    stop = incsubtensors[-1][1]
    x = assignments[-1]['x']
    v = assignments[-1]['v']
    set_instead_of_inc = assignments[-1]['s/i']
    rval = theano.tensor.inc_subtensor(
        x[start:stop],
        v[start:stop],
        set_instead_of_inc=set_instead_of_inc,
        )
    return [rval]


@register_specialize
@register_canonicalize
@local_optimizer()
def local_cut_whole_incsubtensor(node):
    # TODO: template works only for vectors, because of reshape varargs
    match_inc_subtensor = Match('is1', IncSubtensor, idx_list='i1', set_instead_of_inc='s/i')
    template = match_inc_subtensor('x', 'inc_val')

    shape_of = node.fgraph.shape_feature.shape_of

    assignment = template.match(node, {})
    if assignment:
        # print assignment
        i1 = assignment['i1']
        x = assignment['x']
        try:
            x_len = get_scalar_constant_value(shape_of[x][0])
        except NotScalarConstantError:
            # can happen before constant-folding?
            #print "Not a scalar constant??"
            #debugprint(shape_of[x][0])
            #print '-- shape of'
            #debugprint(x)
            return
        inc_val = assignment['inc_val']
        if (len(i1) == 1
                and isinstance(i1[0], slice)
                and i1[0].start == 0
                and i1[0].stop == x_len
                and i1[0].step in (1, None)):
            if assignment['s/i']:
                return [assignment['inc_val']]
            else:
                return [assignment['inc_val'] + x]

