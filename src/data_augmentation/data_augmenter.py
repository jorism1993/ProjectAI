# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:47:34 2018

@author: Joris
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from random import randint
from random import uniform

PATH_TO_BELGIAN_DATA = os.path.join('..','..','data','Belgian','train')

class DataAugmenter(object):
    
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        
        self.image_folders = [name for name in os.listdir(self.path_to_data) 
            if os.path.isdir(os.path.join(self.path_to_data, name))]
        
    def augment_belgian_images(self, operations_per_image = 3):
        
        file_extension = '.ppm'
        
        for folder in self.image_folders:
            path_to_folder = os.path.join(self.path_to_data, folder)
            
            image_list = [image for image in os.listdir(path_to_folder)
                if image.endswith('.ppm')]
            
            for image in image_list:
                im = cv2.imread(os.path.join(self.path_to_data, folder, image))
                im = im.astype(np.float32) / 255
                
                image_name = image.replace('.ppm','')
                
                for i in range(0,operations_per_image):
                    new_im = self.random_gamma_and_rotation(im)
                    
                    path_to_save = os.path.join(self.path_to_data, folder)
                    
                    file_name = image_name + '-'+ str(i) +  file_extension
                    
                    self.save_image(new_im, path_to_save, file_name)
                    
    def delete_all_augmented_images(self):
        
        for folder in self.image_folders:
            path_to_folder = os.path.join(self.path_to_data, folder)
            
            image_list = [image for image in os.listdir(path_to_folder)
                if (image.endswith('.ppm') or image.endswith('.jpg')) and image.count('-') > 0]
            
            for image in image_list:
                os.remove(os.path.join(self.path_to_data, folder, image))
            
                    
    def augment_image(self):

        im = cv2.imread(os.path.join(self.path_to_data,self.image_folders[0])) 
        im = im.astype(np.float32) / 255 
        
        im = self.adjust_gamma(im, gamma = -1.5)
        
        image_rotated_cropped = self.random_rotate_and_crop_image(im)
        
        plt.figure()
        plt.imshow(image_rotated_cropped)
        
        return 
        
    def rotate_and_crop_image(self, im, angle):
        
        image_width = im.shape[0]
        image_height = im.shape[1]
        
        rot_image = self.rotate_image(im, angle)
        
        image_rotated_cropped = self.crop_around_center(
            rot_image,
            *self.largest_rotated_rect(
                image_width,
                image_height,
                math.radians(angle)
            )
        )
            
        return image_rotated_cropped
    
    def random_rotate_and_crop_image(self, im):
        angle = randint(-12, 12)
        
        return self.rotate_and_crop_image(im, angle)
        
    def add_uniform_noise(self, im, noise_factor = 0.15):
        row, col, _ = im.shape
        
        norm = np.random.random((row, col, 1)).astype(np.float32)
        norm = np.concatenate((norm, norm, norm), axis = 2)
        
        norm_img = cv2.addWeighted(im, (1- noise_factor), norm, noise_factor, 0)
        
        return norm_img
    
    def adjust_gamma(self, im, gamma=1.5):
        
        if np.max(im) <= 1.05:
            im *= 255
            im = im.astype(np.int8)

        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")
        
        gamm_image = cv2.LUT(im, table)
        return gamm_image
    
    def random_gamma_and_rotation(self, im):
        
        new_im = copy.copy(im)
        
        gamma = uniform(0.1 , 3.5)
        
        new_im = self.adjust_gamma(new_im, gamma = gamma)
        
        new_im = self.random_rotate_and_crop_image(new_im)
        
        return new_im   

    def save_image(self, im, path, file_name):
        
        cv2.imwrite(os.path.join(path,file_name), im)     
        return
    
    def rotate_image(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """
    
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)
    
        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )
    
        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
    
        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5
    
        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]
    
        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]
    
        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]
    
        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)
    
        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))
    
        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])
    
        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    
        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )
    
        return result
    
    
    def largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
    
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    
        Converted to Python by Aaron Snoswell
        """
    
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi
    
        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    
        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    
        delta = math.pi - alpha - gamma
    
        length = h if (w < h) else w
    
        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)
    
        y = a * math.cos(gamma)
        x = y * math.tan(gamma)
    
        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )
    
    
    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """
    
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    
        if(width > image_size[0]):
            width = image_size[0]
    
        if(height > image_size[1]):
            height = image_size[1]
    
        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)
    
        return image[y1:y2, x1:x2]
    
if __name__ == '__main__':
    DA = DataAugmenter(PATH_TO_BELGIAN_DATA)
    gauss_im = DA.augment_belgian_images()