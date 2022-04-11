# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:42:51 2022




@author: Till
"""

import lib.io as io
import lib.net as net


##import images and labels, get files from the MNIST database
nr_of_trainimages   = 60000 #0 - 60000
nr_of_testimages    = 200   #0 - 10000
labeled_imageS      = io.Lada(nr_of_trainimages, 28,'train-images-idx3-ubyte.gz'\
,'train-labels-idx1-ubyte.gz') #"28" is the size of the quadratic images
    
labeled_imageS_test   = io.Lada(nr_of_testimages, 28,'t10k-images-idx3-ubyte.gz'\
,'t10k-labels-idx1-ubyte.gz') #"28" is the size of the quadratic images    

train_imageS        = labeled_imageS.flattened_images
train_labelS        = labeled_imageS.labels
test_imageS         = labeled_imageS_test.flattened_images
test_labelS         = labeled_imageS_test.labels

    
##create neural net
# instead of the database images from MNIST, you can use any numpy vector to train or test the net
# the size of the first layer (here: 784) has to match the dimension of the training data (here: 28*28 = 784)
# the size of the last layer (here: 10) has to match the number of possible labels (here: 10, corresponding to the numbers 0 - 9)
# you may freely add or remove layers in between (not first or last) or change their size
netz = net.Net([784,16,16,10])


##initialize net with random biases and weights
netz.rnd()


##trainier das netz
# "3" is the number of training epochs, i.e. how many times the net is trained with the same data consequently
for i in range(3): 
    netz.train(train_imageS, train_labelS, 1, 0.01) 
        

##teste das netz!
##explore the variable "result of testing"; row 0-9: activation, row 10: most probable activation, row 11: label
result_of_testing = netz.test(test_imageS, test_labelS)

