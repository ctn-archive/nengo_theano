Basic API
=========

The NEF API operates on two levels.
This page describes the basic high-level interface
for creating neural simulations using the
Neural Engineering Framework (NEF).

The easiest way to interact with the NEF API
is to instantiate a :class:`nef.nef_theano.Network` object,
and call the appropriate methods on that object
to construct ensembles and connect them together.
This API should be sufficient for 90% of models
created using the NEF.

nef.Network
-----------

.. autoclass:: nef.nef_theano.Network
   :members: