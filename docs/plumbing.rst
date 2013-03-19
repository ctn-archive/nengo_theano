============
Advanced API
============

The NEF API operates on two levels.
This page describes the more complicated low-level interface
for creating neural simulations using the
Neural Engineering Framework (NEF).

This API is designed for more experienced
modelers who need more complicated functionality
than is offered by the :class:`nef.nef_theano.Network` class.
This API exposes the underlying objects
that are created by the methods in :class:`nef.nef_theano.Network`,
allowing for more fine-grained control and subclassing.

Nodes
=====

nef.Network
-----------

.. autoclass:: nef.nef_theano.Network
   :noindex:
   :members:

nef.Ensemble
------------

.. autoclass:: nef.nef_theano.Ensemble
   :members:

nef.Input
---------

.. autoclass:: nef.nef_theano.Input
   :members:

nef.SimpleNode
--------------

.. autoclass:: nef.nef_theano.SimpleNode
   :members:

Logging
=======

nef.Probe

.. autoclass:: nef.nef_theano.Probe
   :members:

Connections
===========

nef.Origin
----------

.. autoclass:: nef.nef_theano.Origin
   :members:

nef.EnsembleOrigin
------------------

.. autoclass:: nef.nef_theano.EnsembleOrigin
   :members:

nef.LearnedTermination
----------------------

.. autoclass:: nef.nef_theano.LearnedTermination
   :members:

