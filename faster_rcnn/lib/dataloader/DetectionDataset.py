import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import csv

class DetectionDataset(Dataset):
	def __init__(self, csv_path, root_dir, transform=None, use_superclass=False):
		"""
		Args:
		    csv_file (string): Path to the csv file with annotations.
		    root_dir (string): Directory with all the images.
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
		self.annotations = []
		with open(csv_path, newline='') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')
			for row in csv_reader:
				self.annotations.append(row)

		self.root_dir = root_dir
		self.transform = transform
		self.use_superclass = use_superclass


	def __len__(self):
		return len(self.annotations)


	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.annotations[idx][0])

		n_bboxes = len(self.annotations[idx])
		n_bboxes -= 1 # Don't count the name
		n_bboxes /= 6 # There are 6 information fields per bounding box
		n_bboxes = int(n_bboxes)

		# Change the extension from jp2 to jpg
		img_name = img_name[:-3] + 'jpg'

		image = io.imread(img_name)
		image = np.array(image)
		image = np.moveaxis(image, -1, 0)
		image = np.expand_dims(image, axis=0)

		image_info = np.array([image.shape[2], image.shape[3], 1.5])

		bboxes = np.zeros((1, 20, 5))

		for i in range(n_bboxes):
			bbox = self.annotations[idx][i*6+1: i*6+5]
			class_label = self.annotations[idx][i*6+5]
			superclass_label = self.annotations[idx][i*6+6]

			if self.use_superclass:
				bbox.extend(int(float(superclass_label)))
			else:
				bbox.append(int(float(class_label)))

			bboxes[0, i, :] = bbox

		data = (image, image_info, bboxes, n_bboxes)

		return data


	def show_example(index=0):
		fig = plt.figure()
		ax = plt.subplot(1, 1, 1)

		sample = detection_dataset[5]

		plt.tight_layout()
		im = sample[0]
		im = np.squeeze(im, axis=0)

		bboxes = sample[2][0]
		n_bboxes = sample[3]
		for i in range(n_bboxes):	
			bbox = bboxes[i]
			bbox = bbox.reshape(-1)

			rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
			ax.add_patch(rect)

		im = np.moveaxis(im, 0, -1)	
		ax.imshow(im)
		plt.show()