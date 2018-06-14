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
from PIL import Image


width = 600
height = 450


data_transform = transforms.Compose([
        transforms.Resize((height,width))
    ])



class DetectionDataset(Dataset):

	def __init__(self, root_dir, transform=data_transform, use_superclass=False, use_data='all'):
		"""
		Args:
		    root_dir (string): Directory with all the images and annotation files
		    transform (callable, optional): Optional transform to be applied on a sample.
		    use_superclass (boolean, optional): use superclasses for the Belgian dataset rather than normal classes
		    use_data (string, optional): specify which data source to use ('belgian', 'german', 'all')
		"""

		self.annotations = []
		self.root_dir = root_dir
		self.transform = transform
		self.use_superclass = use_superclass

		belgian = use_data == 'belgian' or use_data == 'all'
		german = use_data == 'german' or use_data == 'all'

		if belgian:
			self.load_csv_to_annotation_list('belgian')

		if german:
			self.load_csv_to_annotation_list('german')


	def load_csv_to_annotation_list(self, dataset):
		csv_path = self.root_dir + dataset + '_annotations.txt'
		folder = dataset + '/'
		if dataset == 'belgian':
			extension = 'jpg'
		if dataset == 'german':
			extension = 'ppm'

		with open(csv_path, newline='') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')

			for row in csv_reader:
				row[0] = folder + row[0]
				row[0] = row[0][:-3] + extension
				self.annotations.append(row)


	def __len__(self):
		return len(self.annotations)


	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.annotations[idx][0])

		n_bboxes = len(self.annotations[idx])
		n_bboxes -= 1 # Don't count the name
		n_bboxes /= 6 # There are 6 information fields per bounding box
		n_bboxes = int(n_bboxes)
		
		try:
		  image = io.imread(img_name)
		except Exception as e:
		  return (None, None, None, None)

		image = np.array(image)
		img_pil = Image.fromarray(image)

		image_info = np.array([image.shape[0], image.shape[1], 1.5])

		bboxes = np.zeros((1, 20, 5))

		for i in range(n_bboxes):
			bbox = self.annotations[idx][i*6+1: i*6+5]

			bbox[0] = float(bbox[0]) * float(width / image_info[1])
			bbox[2] = float(bbox[2]) * float(width / image_info[1])
			bbox[1] = float(bbox[1]) * float(height / image_info[0])
			bbox[3] = float(bbox[3]) * float(height / image_info[0])

			class_label = self.annotations[idx][i*6+5]
			superclass_label = self.annotations[idx][i*6+6]

			if self.use_superclass:
				bbox.append(int(float(superclass_label)))
			else:
				bbox.append(int(float(class_label)))

			bboxes[0, i, :] = bbox


		image = self.transform(img_pil)
		image = np.array(image)
		image = np.moveaxis(image, -1, 0)
		image = np.expand_dims(image, axis=0)

		data = (image, image_info, bboxes, n_bboxes)

		return data


	def get_file_path(self, idx):
		return self.annotations[idx][0]


	def show_example(self, index=0):
		fig = plt.figure()
		ax = plt.subplot(1, 1, 1)

		sample = self[0]

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



# detection_dataset = DetectionDataset(root_dir='../data/detection_data/', use_data='all')
# print('Number of pictures:', len(detection_dataset))
# detection_dataset.show_example()