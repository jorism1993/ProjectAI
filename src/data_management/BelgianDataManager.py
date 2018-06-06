# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:45:46 2018

@author: Joris
"""

import torch
from torchvision import transforms, datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

class BelgianDataManager(object):
    
    def __init__(self,PATH_TO_BELGIAN_DATA):

        self.PATH_TO_BELGIAN_DATA = PATH_TO_BELGIAN_DATA        
    
    def set_transforms(self, resize=(100,100) ):
        """ Function to set the transforms 
            As input it takes an optional argument 'resize', which is a tuple
            of the required size to resize to. """
            
        data_transforms = {
            'train': transforms.Compose([transforms.Resize(resize),transforms.ToTensor(),]),
            'test': transforms.Compose([transforms.Resize(resize),transforms.ToTensor(),])  
            }
        return data_transforms
    
    def retrieve_image_datasets(self):
        """ Function to retrieve the image datasets for training and test images """
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.PATH_TO_BELGIAN_DATA,x),
                self.data_transforms[x]) for x in ['train','test']}
        return image_datasets
    
    def set_dataloaders(self):
        """Function that defines the data loader object in a dict """
        
        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4,
                shuffle=True, num_workers=4) for x in ['train','test']}
        return dataloaders

    def get_dataset_size(self):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train','test']}
        return dataset_sizes
    
    def get_class_names(self):
        class_names = self.image_datasets['train'].classes
        return class_names
    
    def load_data(self, resize=(100,100) ):
        """ Function that can be called that loads the data """
        
        self.data_transforms = self.set_transforms(resize=resize)
        self.image_datasets = self.retrieve_image_datasets()
        dataloaders = self.set_dataloaders()
        dataloaders['train'].num_workers = 0
        dataloaders['test'].num_workers = 0
        
        return dataloaders
    
    def plot_a_few_images(self):
        """ Plot a few images from the training set """
        
        self.data_transforms = self.set_transforms()
        self.image_datasets = self.retrieve_image_datasets()
        dataloaders = self.set_dataloaders()
        dataloaders['train'].num_workers = 0
        dataloaders['test'].num_workers = 0
        
        class_names = self.get_class_names()
        
        def imshow(inp, title=None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
        
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        
        imshow(out, title=[class_names[x] for x in classes])
        return
    
if __name__ == '__main__':
    
    PATH_TO_BELGIAN_DATA = os.path.join('..','..','data','Belgian')
    BelgianData = BelgianDataManager(PATH_TO_BELGIAN_DATA)
    
    BelgianData.plot_a_few_images()
    data = BelgianData.load_data(resize = (400,400))