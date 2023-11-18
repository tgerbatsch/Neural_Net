# Neural_Net
## introduction
hi there!\
this is a neural net class implemented in python3 using only numpy as library \
The purpose of the net is to classify/recognize written numbers [1 - 9] \
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


## Result
Note that the output of the demo.py is not especially graphically entertaining.
When the process is finished, a variable called "result_of_testing" will be calculated.
You have to inspect that variable "result_of_testing" manually.
In the rows 0-9 you will find an indication of how likely the networks predicts the corresponding number.
in row 10, the highest "probabaility" will be indicated.
In row 11, you will find the label of the number (so what the picture is actually supposed to show ...).
So, in order to see that the network is working, you have to do the following:

1)look at the label in row 11, it will be an integer [0, ...,9]
2)look at the entry in the corresponding row
3)compare that entry with the entry in row 10 (which will tell you if the network prediction matches the label)
