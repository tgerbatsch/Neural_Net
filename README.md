# Neural_Net
## introduction
hi there!\
this is a neural net class implemented in python3 using only numpy as library \
this particular neural net is based on the explanation from the "3blue 1brown" youtube video on neural networks\
without using their coding examples. The tensor notation is the same as in the video.\
run demo.py for a first impression. It may take a couple of minutes\
preferably run the python script in an IDE to see the different variables\
## about the net
the net is stored as an instance of the class "Net" within lib/net \
Net.knots are the knots of the net, i.e. the current activation\
Net.weights are weights between the net, i.e. the essential information of the network

## Purpose
This is mainly a fun project to understand the tensor operations inside a neural network. The tensor notation is the same as in the youtube video on neural networks of "3blue 1brown".
The main technical finding may be in the lib/net.grad() method where the gradient is calculated via backpropagation.
