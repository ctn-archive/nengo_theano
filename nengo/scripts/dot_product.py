from .. import nef_theano as nef

def make_dot(net, name, neurons, dimensions1, dimensions2, mode='spiking'):
    """A function that makes a subnetwork that calculates the dot product
    between a matrix and a vector, and adds it to the network.

    :param Network net: the network to add to
    :param string name: the name of the dot product subnetwork
    :param int neurons: the number of neurons per ensemble
    :param int dimensions1: number of rows in matrix
    :param int dimensions2: number of columns in matrix and size of vector
    """
    # define multiplication function for ensemble connections
    def product(x): return x[0] * x[1]
    
    # create subnetwork for dot product
    dot = net.make_subnetwork(name) 
    
    #input relay for matrix dimensions1 x dimensions2
    dot.make('input_matrix', neurons=1, 
        dimensions=dimensions1*dimensions2, mode='direct') 
    #input relay for vector 1 x dimensions2
    dot.make('input_vector', neurons=1, 
        dimensions=dimensions2, mode='direct') 
    # create output relay
    dot.make('output', neurons=1, 
        dimensions=dimensions1, mode='direct')

    # create one network for each row of dot multiplication
    for d1 in range(dimensions1):

        dot.make(name='row_mult%d'%d1, neurons=neurons, 
            array_size=dimensions2, dimensions=2, radius=1.5,
            encoders=[[1,1,],[1,-1],[-1,1],[-1,-1]], mode=mode)

        # connect up the row_mult network to the input 
        # pstc = 0 so no delay introduced by relay populations
        dot.connect('input_matrix', 'row_mult%d'%d1, 
            index_pre=range(d1*dimensions2, d1*dimensions2+dimensions2), 
            index_post=range(0, dimensions2*2, 2), pstc=0)
        dot.connect('input_vector', 'row_mult%d'%d1, 
            index_post=range(1, dimensions2*2+1, 2), pstc=0)
        
        # set up connection weights for proper summation at output
        dot.connect('row_mult%d'%d1, 'output', 
            index_post=d1, func=product)
    
def test_dot():

    net = nef.Network('Dot product test')

    import numpy as np
    matrix = np.array([[.1, .2],[.3, .4], [.5, .6]])
    vector = np.array([.4, .5])
    net.make_input('input_matrix', value=matrix.flatten())
    net.make_input('input_vector', value=vector)

    make_dot(net, 'dot_product', neurons=1, 
        dimensions1=3, dimensions2=2, mode='direct')

    net.connect('input_matrix', 'dot_product.input_matrix')
    net.connect('input_vector', 'dot_product.input_vector')

    im_probe = net.make_probe('dot_product.input_matrix')
    iv_probe = net.make_probe('dot_product.input_vector')
    row_probe = net.make_probe('dot_product.row_mult0')
    dot_probe = net.make_probe('dot_product.output')

    net.run(1)

    import matplotlib.pyplot as plt
    plt.ion(); plt.close()
    plt.subplot(311); plt.title('matrix input')
    plt.plot(im_probe.get_data())
    plt.subplot(312); plt.title('vector input')
    plt.plot(iv_probe.get_data())
    a = 'answer: ', np.dot(matrix, vector)
    plt.subplot(313); plt.title(a)
    plt.plot(dot_probe.get_data())
    plt.tight_layout()

#test_dot()
