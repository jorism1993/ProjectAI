import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd



class DetectionDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Args:
		    csv_file (string): Path to the csv file with annotations.
		    root_dir (string): Directory with all the images.
		    transform (callable, optional): Optional transform to be applied
		        on a sample.
		"""
		self.annotations = pd.read_csv(csv_file, sep=';')
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])

		# Change the extension from jp2 to jpg
		img_name = img_name[:-3] + 'jpg'

		image = io.imread(img_name)

		bbox = self.annotations.iloc[idx, 1:5].as_matrix()
		bbox = bbox.astype('float').reshape(-1, 4)

		sample = (image, bbox)

		if self.transform:
		    sample = self.transform(sample)

		return sample

	def show_example(index=0):
		fig = plt.figure()
		ax = plt.subplot(1, 1, 1)

		sample = detection_dataset[6]

		plt.tight_layout()
		im = sample[0]
		imsize = im.shape[0:2]
		bbox = sample[1]
		bbox = bbox.reshape(-1)

		rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)


		ax.imshow(im)
		plt.show()



detection_dataset = DetectionDataset(csv_file='../data/detection_data/annotations_test.txt', root_dir='../detection_data/')
detection_dataset.show_example()