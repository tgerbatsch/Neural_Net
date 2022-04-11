# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:38:58 2022

@author: Till
"""

### "io" is meant for data import, for the moment this will be mainly the mnist data
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 23:44:31 2022

@author: Till
"""
import gzip
import numpy as np
from lib.func import env_pot2



##labeled data
class Lada(object):
    
    
    ##constructor
    def __init__(self, nr= 100,image_size = 28 \
            ,path_images ='train-images-idx3-ubyte.gz'\
            , path_labels= 'train-labels-idx1-ubyte.gz'\
            , **kwargs):
    
        dat = self.get(nr, image_size, path_images, path_labels)
        images = dat[0]
        labels = dat[1] 
        
        self.crt(images, labels, **kwargs)
   
    
    ##create data structure with given data = images & labels
    def crt(self, images, labels, **kwargs ):
        ##check if images and labels fit, if not, truncate
        check =( np.shape(images)[0] == np.shape(labels)[0] )
        
        self.nr_images= nr_images = len(images[:,0,0])
        self.image_size_x = image_size_x = len(images[0,:,0])
        self.image_size_y = image_size_y = len(images[0,0,:])
        self.images = images #normalize
        self.labels = labels
        
        ##flatten the images for neural net input
        flattened_images = np.zeros((nr_images, image_size_x*image_size_y))
        for i in range(nr_images):
            flattened_image = images[i,:,:].flatten()
            flattened_images[i] = flattened_image
         
        #self.flattened_images = flattened_images    
        
        ## normalize images
        ##find min and max powers of 2 in the data, include possible negative values
        if 'min_val' in kwargs:
            min_val = kwargs['min_val']
        else:
            min_val = np.amin (flattened_images[:,:]) ##min value
        ## jetzt hol noch irgendwie geschickt die potenz von 2 raus ...
            min_val = env_pot2(min_val)
            #print (min_val)

        
        if 'max_val' in kwargs:
            max_val = kwargs['max_val']
        else: 
            max_val = np.amax (flattened_images[:,:]) ##max value
            max_val = env_pot2(max_val)
            #print (max_val)
        
        print('images normalized from values between ', min_val, ' and ', max_val , ' to [0, 1]')    
        self.flattened_images = (flattened_images + min_val) / (max_val - min_val) #normalize [0,1]


    ##import data from pathes        
    def get(self, nr= 100,image_size = 28 \
            ,path_images ='../train-images-idx3-ubyte.gz'\
            , path_labels= '../train-labels-idx1-ubyte.gz'):
        
        num_images = nr
        
        ##import images
        f = gzip.open(path_images,'r')
        f.read(16) #skip the first 16 bytes, mnist hat header lines
        #nach .read() wird an der Stelle weiter Daten ausgelesen?
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        images= np.asarray(data[0:num_images]).squeeze()

        ##import labels
        f = gzip.open(path_labels,'r')
        f.read(8) #skip the first 8 bytes, mnist header oder so
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8, count= num_images).astype(np.int64)
        
        return images, labels
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        