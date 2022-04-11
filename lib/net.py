# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:46:13 2022

@author: Till
"""

###neural net class and its functions

##libraries
import numpy as np

##
    
##cost function
class Cf(object):
    
    ##mean square error and its derivative
    @staticmethod
    def mean_square (ist, soll):
        return (ist - soll)*(ist - soll), 2*(ist - soll) 


##the net
class Net(object):
    
    ##constructor, lys = "layers"
    def __init__(self, lys = [784, 16, 16, 10], af = 'sigmoid' ):
          
        ##initialize knots und weights und biases
        self.number_of_layers = number_of_layers = len(lys)
        lys = np.array(lys)
        knots = []
        weights=[]
        biases =[]
        for i in range(number_of_layers - 1):
            #name_of_layer = "a_upper" + str(i)
            ##knoten sind eine liste von arrays. jdr array hat die groesse des entsprechenden layers
            knots.append(np.zeros(lys[i])) 
            biases.append(np.zeros(lys[i+1])) 
            
            ##weights sind eine Liste aus Matrizen
            weights.append( np.zeros((lys[i+1], lys[i])) )
        
            
        ##initialisiere knots letzte layer
        knots.append(np.zeros(lys[number_of_layers -1]))
        #biases.append(np.zeros(lys[i])) 
        
        self.lys     = lys
        self.knots   = knots
        self.weights = weights
        self.biases  = biases
        
        ##initialisiere die aktivierungsfunktion und deren ableitung
        self.activation_func = Af().f(af) #funktion und ableitung
        #self.activation_func = activation_func[0] #TypeError: 'function' object is not subscriptable
        #self.activation_deriv = activation_func[1]
        
        
    ##assign random weights to the net in [-1, +1]
    def rnd(self):
            
        number_of_layers = self.number_of_layers
        lys = self.lys
        weights=[]
        biases =[]
        for i in range(number_of_layers - 1):
            #name_of_layer = "a_upper" + str(i)
            ##knoten sind eine liste von arrays. jdr array hat die groesse des entsprechenden layers
            rand_bias  = np.random.rand(lys[i+1]) * 2 - 1
            biases.append(rand_bias) 
            
            ##weights sind eine Liste aus Matrizen
            rand_weights = np.random.rand(lys[i+1], lys[i]) * 2 - 1
            weights.append( rand_weights )
        
            
        ##initialisiere knots letzte layer
        #rand_bias  = np.random.rand(lys[i])
        #biases.append(np.zeros(lys[i])) 
        
        self.weights = weights
        self.biases  = biases
                

    ##"propagation": calculate knots of given net as a function of its activation
    def prop(self, activation):
        
        ## the first layer of knots equals the activation
        self.knots[0] = activation
        for i in range(self.number_of_layers -1):
            z = np.matmul (self.weights[i], self.knots[i]) + self.biases[i]
            self.knots[i + 1] = self.activation_func(z)[0]
            
            
    ##test data: return an array with the actual data and the label gegenüber gestellt
    def test(self, activationS, labelS):
        
        ##nr of test entries
        nr_data = np.size(labelS)
        
        ##build an array with the activation of the last layer knots, the most probable knot and the label
        return_value = np.zeros((nr_data, self.lys[self.number_of_layers - 1] + 2))        
        for i in range(nr_data):
            self.prop(activationS[i,:])
            for j in range ( ( self.lys[self.number_of_layers -1] ) ):
                return_value[i,j] = self.knots[self.number_of_layers -1][j]        
            max_activation = np.amax(return_value[i,:])
            return_value[i, self.lys[self.number_of_layers -1]] = max_activation
            return_value[i, self.lys[self.number_of_layers -1] + 1] = labelS[i]
        
        return return_value    
            

    ##"backpropagation": calculate the gradient of given cost function
    ##of a single activation and a single label
    def grad(self, activation, label, costfunc = Cf().mean_square):
        
        ##help function for easier notation
        ##wL is weight Matrix, (jacobi) is wL.T; aLk_min1 is a vector
        ##returns array of z_j
        def zj_(aLk_min1, wL, bL):
            return np.matmul(wL , aLk_min1) + bL
            #return wL * aLk_min1 + bL
            
        
        ##calculate the derivatives of weights ans biases
        ## "z" is a vector with index j , aLk is a vector with different indizees
        ##return da_dw
        ##returns array of daj_dwjk
        def daj_dwjk(aLk_min1, activation_func, z):
            daf = activation_func(z)[1] ##derivative of activation func
            return np.outer(aLk_min1, daf)
            
        
        ##return da_db
        ## "z" is a vector with index j
        ##returns array of daj_dbj
        def daj_dbj( activation_func, z):
            return activation_func(z)[1] 
    

        ##return da_damin1
        ## "z" is a vector with index j, wL_jk ist die weightmatrix zwischen Schicht L und L-1
        ##returns matrix of daj_dak
        ##fuer die Kettenregel gehen wir spaeter von schicht n(j) nach n-1(k), also ist es eine k mal j matrix
        ## ... hat also die dimensionen der tranponierten gewichtsmatrix
        def da_da_min1(wL_jk, activation_func, z):
            return np.multiply(wL_jk.T, activation_func(z)[1] )
        
        ##propagate the activation through the network
        self.prop(activation)
        
        number_of_layers = self.number_of_layers
        self.grad_knots     =[None] * number_of_layers      ##corresponds to the a`s on 3blue one brown
        self.grad_weights   =[None] * (number_of_layers -1) ##corresponds to the w`s on 3blue one brown
        self.grad_biases    =[None] * (number_of_layers -1) ##corresponds to the b`s on 3blue one brown
        ist = self.knots[number_of_layers - 1]
        layers = self.lys
        size_of_last_layer = layers[number_of_layers -1]
        soll = np.fromfunction(lambda x: 1*(x==label), (size_of_last_layer,)) 
        #cost = np.sum( Cf().mean_square(ist, soll)[0] )              
        #costf = ( Cf().mean_square(ist, soll) ) ##cost and its derivative
        costf = ( costfunc(ist, soll) ) ##cost and its derivative
                
        
        ##calculate derivative of knots of last layer
        self.grad_knots[number_of_layers - 1] = costf[1] ## dC/daL
        
        ##backpropagation
        activation_function = self.activation_func
        for i in range (number_of_layers -2,-1,-1):
            
            ##dC_da
            dC_a = self.grad_knots[i+1]
            
            ##biases of layer i
            bL = self.biases[i]
            
            ##weights from layer i-1 to layer i
            wL = self.weights[i]
            
            ##activations of layer i -1
            aL_min1 = self.knots[i] ##note knot[i] is activated through weight i-1
            
            ##zL_j = ...
            zL_j = zj_(aL_min1, wL, bL)
            
            ##da/dw = ... # derivative tensor of a with resprect to w
            da_dw_i = daj_dwjk(aL_min1, activation_function, zL_j)##weight gradient of current layer
            
            ##da/db = ...
            da_db_i = daj_dbj(activation_function, zL_j)
            #self.grad_bias.insert(i, grad_bias_i)
            
            ##dC/dw = dC/da * da/dw
            dC_dw_i = (np.multiply(da_dw_i ,dC_a)).T
            self.grad_weights[i] = dC_dw_i
            
            ##dC/db = dC/da * da/db
            dC_db_i = (np.multiply(da_db_i ,dC_a))
            self.grad_biases[i] =  dC_db_i
            
            ##da/da_min1 = ...
            dada_min1 = da_da_min1(wL, activation_function, zL_j)
            
            ##dC/da_min1(da/da_min1) = ... 
            dC_da_min1 = np.matmul(dada_min1, dC_a)
            self.grad_knots[i]= dC_da_min1

        
    ## "multigradient" calculate the gradient of n activations and its corresponding n labels             
    def ngrad(self, activationS, labelS, costfunc = Cf().mean_square):
        
        ##activations n labels are supposed to be of the form (N x Hoehe x Breite)
        ## and (N) respectively
        if np.shape(activationS)[0] != np.shape(labelS)[0]:
            raise 'data not of appropriate size'
        else:
            nr_of_images = np.shape(activationS)[0]
        
        
        ##calculate gradient of first entry to initialize grad_weight, grad_biases
        activation = activationS[0]
        label      = labelS[0]
        self.grad(activation, label, costfunc)
        grad_weights = self.grad_weights
        grad_biases  = self.grad_biases
        grad_knots   = self.grad_knots
        
        
        ##calculate the gradients of the rest of the data and sum it up
        for i in range (1,nr_of_images):
            activation = activationS[i]
            label      = labelS[i]
            self.grad(activation, label, costfunc)
            for j in range(len(grad_weights)):
                grad_weights[j] += self.grad_weights[j]
                
            for j in range(len(grad_biases)):
                grad_biases[j]  += self.grad_biases[j]
                
            for j in range(len(grad_knots)):
                grad_knots[j] += self.grad_knots[j]
            
            
        ##normalize
        for j in range(len(grad_weights)):
            grad_weights[j] /= nr_of_images
            
        for j in range(len(grad_biases)):
            grad_biases[j]  /= nr_of_images
            
        for j in range(len(grad_knots)):
            grad_knots[j]   /= nr_of_images
 

    ##move along the negative gradient with step length alpha
    ## the gradient (self.grad_weights etc. ...) has to be initialized                                           
    def step_grad(self, alpha = 0.1):
        
        weights     = self.weights
        grad_weights= self.grad_weights
        biases      = self.biases
        grad_biases = self.grad_biases
        
        
        for j in range(len(grad_weights)):
            self.weights[j] = weights[j] - alpha * grad_weights[j]
        
            
        for j in range(len(grad_biases)):
            self.grad_biases[j] = biases[j] - alpha * grad_biases[j]
    
    
    ##train the net with a given set of data, labels
    def train(self, activationS, labelS, bundlesize, steplength = 0.1\
              ,costfunc = Cf().mean_square ):
        
        ##check, whether the bundlesize fits the number of data sets
        if np.shape(activationS)[0] != np.shape(labelS)[0]:
            raise 'data not of appropriate size'
        else:
            nr_of_images = np.shape(activationS)[0]
        rest = nr_of_images % bundlesize 
        
        ##set step counter for appropriate data use
        counter = 0
        
        
        ##move some steps of gradient of single images
        for i in range(rest):
            activation = activationS[counter]
            label      = labelS[counter]
            
            ##calculate knots
            #self.prop(activation)
            
            ##calculate gradient
            self.grad(activation, label, costfunc)
    
            ##move step of given length towards the negative gradient
            self.step_grad(steplength)
            
            ##count place at which image i am
            counter += 1
        
            
        ##move steps of bundled gradient
        for i in range(int(nr_of_images/bundlesize - rest)):
            activations_bundle = activationS[counter:counter + bundlesize]
            labels_bundle      = labelS[counter:counter + bundlesize]
            
            ##calculate gradient 
            self.ngrad(activations_bundle, labels_bundle)
            
            ##move step of given length towards the negative gradient
            self.step_grad(steplength)
            
            ##count place at which image i am
            counter += bundlesize
       
        
       
##activation function
class Af(object):
    
    #set of available functions
    
    #return sigmoid and its derivative
    def sigmoid(x):
        #if x > 128 : return 1 #ambigous ...
        #if x < -128: return 0
        x = (x>= 16)*16 + (x>-16)*(x<16)*x + (x<= -16)*(-16)
        #return_val_function   = 1*(x>128) + (x>-128)*(x<=128)*1/(1 + np.exp(-x)) ##immer noch runtime warnings u damit nans
        #return_val_derivative = 1*(x>128) + (x>-128)*(x<=128)*(np.exp(-x)) / ( (1 + np.exp(-x))**2 )
        #return return_val_function, return_val_derivative
        return [ 1/(1 + np.exp(-x)), (np.exp(-x)) / ( (1 + np.exp(-x))**2 ) ]
    
    
    funcdic = { 'sigmoid':sigmoid }
    
    #method, that returns the desired function
    def f(self, func = 'sigmoid'):     
        return self.funcdic[func]
        
    
 
   
   