from nef.templates import basalganglia
import nef.nef_theano as nef

D=5
net=nef.Network('Basal Ganglia') #Create the network object

#Make a basal ganglia model with 50 neurons per action
basalganglia.make_basal_ganglia(net, dimensions=D, neurons=50)  

net.run(1)
