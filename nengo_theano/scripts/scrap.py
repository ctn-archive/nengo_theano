import nengo_theano as nef

net = nef.Network('Scrap')

import math
net.make_input('input', value=1, zero_after_time=5)#math.sin)
net.make_input('input2', value=1)#math.sin)

net.make('pop', neurons=100, dimensions=1, intercept=[.05,1])
net.make('pop2', neurons=100, dimensions=1)

inhib_matrix = [[-1]] * 100

net.connect('input', 'pop')
net.connect('input2', 'pop2')
net.connect('pop', 'pop2', transform=inhib_matrix, pstc=1)

im_probe = net.make_probe('input')
pop_probe = net.make_probe('pop')
pop2_probe = net.make_probe('pop2')

net.run(18)

import matplotlib.pyplot as plt
plt.plot(im_probe.get_data())
plt.plot(pop_probe.get_data())
plt.plot(pop2_probe.get_data())
plt.legend(['input', 'pop', 'pop2'])
plt.tight_layout()
plt.show()
