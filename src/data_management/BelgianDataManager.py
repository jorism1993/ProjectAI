# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:46:19 2018

@author: Joris
"""
import os
import csv
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

class BelgianDataManager(object):
    
    def __init__(self,PATH_TO_BELGIAN_DATA):
        
        self.path_to_data = PATH_TO_BELGIAN_DATA
        self.folders = self.load_folders()
        
        self.path_to_images = []
        self.image_bounding_box = []
        self.image_class = []
        
        for folder in self.folders:
            self.load_images_from_folder(folder)   
                    
    def load_folders(self):
        """ Function that returns a list of paths to all the folders containing images
        """
        # Load the list of paths, removing the README
        paths = os.listdir(self.path_to_data)
        paths = [path for path in paths if os.path.isdir(os.path.join(self.path_to_data,path))]
        return paths
    
    def load_images_from_folder(self,folder_name):
        """ Given a name of a folder, append a list of paths to all the images
        """
        
        # Define the file_name, this appears to be GT-XXXX.csv
        file_name = "GT-" + folder_name + ".csv"
        path_to_csv = os.path.join(self.path_to_data,folder_name,file_name)
        
        # Open the csv file
        with open (path_to_csv,'r') as csvfile:
            
            file_info = csv.reader(csvfile, delimiter = ';')
            next(file_info, None) # Skip the headers lien
            
            for row in file_info:
                image_file = row[0]
                
                x1 = float(row[3])
                y1 = float(row[4])
                x2 = float(row[5])
                y2 = float(row[6])
                
                classID = row[7]
                
                self.path_to_images.append(os.path.join(self.path_to_data,folder_name,image_file))
                self.image_bounding_box.append(((x1,y1),(x2,y2)))
                self.image_class.append(classID)
        return
    
    def load_image_into_array(self, path_to_image, bounding_box_coordinates, resize = True, resize_shape = [100,100]):
        """ Load an images into a numpy array, given a path to an image
            In the case of a resize, also specify new bounding box coordinates
        """        
        if resize:
            with open(path_to_image,'r+b') as f:
                with Image.open(f) as image:
                    height, width = image.size
                    
                    # Resize the image and convert it to a numpy array
                    resized_img = resizeimage.resize_cover(image, resize_shape, validate=False)
                    imarray = np.array(resized_img)
                    
                    # New coordinates of the bounding box
                    x1 = float(bounding_box_coordinates[0][0]) * (resize_shape[0] / width)
                    y1 = float(bounding_box_coordinates[0][1]) * (resize_shape[1] / height)
                    x2 = float(bounding_box_coordinates[1][0]) * (resize_shape[0] / width)
                    y2 = float(bounding_box_coordinates[1][1]) * (resize_shape[1] / height)
                    new_coordinates = ((round(x1),round(y1)),(round(x2),round(y2)))
                    
        else:
            with open(path_to_image,'r+b') as f:
                with Image.open(f) as image:                   
                    imarray = np.array(image)
                    new_coordinates = bounding_box_coordinates
                    
        return imarray, new_coordinates
    
    def load_all_images_and_labels(self,resize = True, resize_shape = [100,100]):
        """ Load all images into a numpy array
        """
        
        images_list = []
        resized_boxes = []
        
        for i,path in enumerate(self.path_to_images):
            
            # Retrieve image in numpy array format
            bounding_box = self.image_bounding_box[i]
            imarray, bounding_box = self.load_image_into_array(path, bounding_box, resize=resize, resize_shape=resize_shape)
            resized_boxes.append(bounding_box)
            
            # Resize by 255
            imarray = imarray / 255.0
            
            images_list.append(imarray)
            
        immatrix = np.stack(images_list,axis=0)
        return immatrix, self.image_class, resized_boxes     
            
    def plot_images(self):
        """ Plot 5 random images with bounding box, to show it works """
        fig = plt.figure()
            
        for i in range(5):
            # Select idx of random image
            idx = random.randrange(0,len(self.path_to_images))
            
            # Retrieve path and bounding box
            path_to_img = self.path_to_images[idx]
            bounding_box = self.image_bounding_box[idx]
            
            # Retrieve bounding box and box width, height
            imarray, bounding_box = self.load_image_into_array(path_to_img, bounding_box, resize=True)
            box_width = bounding_box[1][0] - bounding_box[0][0] 
            box_height = bounding_box[1][1] - bounding_box[0][1]
            
            # Plot
            ax = fig.add_subplot(1,5,i+1)
            ax.imshow(imarray)
            ax.add_patch(matplotlib.patches.Rectangle(bounding_box[0],box_height,box_width,linewidth = 1, edgecolor='r',fill=False))


if __name__ == '__main__':
    
    PATH_TO_BELGIAN_DATA = os.path.join('..','..','data','Belgian','Training')
    BelgianData = BelgianDataManager(PATH_TO_BELGIAN_DATA)
    
    immatrix, labels, boxes = BelgianData.load_all_images_and_labels(resize = True, resize_shape = [100,100])