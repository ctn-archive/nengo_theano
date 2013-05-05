import theano
import numpy as np

def compute_transform(dim_pre, dim_post, array_size, weight=1,
                      index_pre=None, index_post=None, transform=None):
    """Helper function used by :func:`nef.Network.connect()` to create
    the `dim_post` by `dim_pre` transform matrix.

    Values are either 0 or *weight*. *index_pre* and *index_post*
    are used to determine which values are non-zero, and indicate
    which dimensions of the pre-synaptic ensemble should be routed
    to which dimensions of the post-synaptic ensemble.

    :param int dim_pre: first dimension of transform matrix
    :param int dim_post: second dimension of transform matrix
    :param int array_size: size of the network array
    :param float weight: the non-zero value to put into the matrix
    :param index_pre: the indexes of the pre-synaptic dimensions to use
    :type index_pre: list of integers or a single integer
    :param index_post:
        the indexes of the post-synaptic dimensions to use
    :type index_post: list of integers or a single integer
    :returns:
        a two-dimensional transform matrix performing
        the requested routing

    """

    if transform is None:
        # create a matrix of zeros
        transform = [[0] * dim_pre for i in range(dim_post * array_size)]

        # default index_pre/post lists set up *weight* value
        # on diagonal of transform

        # if dim_post * array_size != dim_pre,
        # then values wrap around when edge hit
        if index_pre is None:
            index_pre = range(dim_pre)
        elif isinstance(index_pre, int):
            index_pre = [index_pre]
        if index_post is None:
            index_post = range(dim_post * array_size)
        elif isinstance(index_post, int):
            index_post = [index_post]

        for i in range(max(len(index_pre), len(index_post))):
            pre = index_pre[i % len(index_pre)]
            post = index_post[i % len(index_post)]
            transform[post][pre] = weight

    transform = np.array(transform)

    # reformulate to account for post.array_size
    if transform.shape == (dim_post * array_size, dim_pre):
        array_transform = [[[0] * dim_pre for i in range(dim_post)]
                           for j in range(array_size)]

        for i in range(array_size):
            for j in range(dim_post):
                array_transform[i][j] = transform[i * dim_post + j]

        transform = array_transform

    return transform


class Case1(theano.Op):
    """
    XXX doc: see network.py for why this is called case1
    """
    def __init__(self, shape4):
        self.shape4 = shape4

    def __hash__(self):
        return hash((type(self), self.shape4))

    def __eq__(self, other):
        return type(self) == type(other) and self.shape4 == other.shape4

    def make_node(self, transform, pre_output):
        # TODO: more accurate broadcasting output pattern
        return theano.Apply(self,
                map(theano.tensor.as_tensor_variable, [transform, pre_output]),
                [theano.tensor.matrix()])

    def perform(self, node, inputs, outstor):
        transform, pre_output = inputs
        # TODO: implement with np.dot
        encoded_output = (transform.reshape(self.shape4)
            * pre_output.reshape(self.shape4[2:]))

        # sum the contribution from all pre neurons
        # for each post neuron
        encoded_output = np.sum(encoded_output, axis=3)
        # sum the contribution from each of the
        # pre arrays for each post neuron
        encoded_output = np.sum(encoded_output, axis=2)
        # reshape to get rid of the extra dimension
        encoded_output.shape = self.shape4[:2]
        outstor[0][0] = encoded_output


class Case2(theano.Op):
    """
    XXX doc: see network.py for why this is called case2
    """
    def __init__(self, shape4):
        self.shape4 = shape4

    def __hash__(self):
        return hash((type(self), self.shape4))

    def __eq__(self, other):
        return type(self) == type(other) and self.shape4 == other.shape4

    def make_node(self, transform, pre_output):
        # TODO: more accurate broadcasting output pattern
        return theano.Apply(self,
                map(theano.tensor.as_tensor_variable, [transform, pre_output]),
                [theano.tensor.matrix()])

    def perform(self, node, inputs, outstor):
        transform, pre_output = inputs
        post_array_size, post_neurons_num = self.shape4[:2]
        pre_array_size, pre_neurons_num = self.shape4[2:]

        encoded_output = np.zeros((post_array_size, post_neurons_num),
                dtype=node.outputs[0].dtype)
        for ii in xrange(post_neurons_num):
            encoded_output[:, ii] = np.dot(transform[:, ii], pre_output)
        outstor[0][0] = encoded_output
