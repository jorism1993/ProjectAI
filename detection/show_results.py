from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform

root_dir = '../data/detection_data/belgian_resized/'
image_to_bbox_dict = {}

file = open('3resultaten.txt', 'r') 
for line in file: 

	split_line = line.split(' ')
	bbox = split_line[-4:]
	bbox[-1] = bbox[-1][:-1]
	image = split_line[0]

	if image in image_to_bbox_dict:
		image_to_bbox_dict[image].append(bbox)
	else:
		image_to_bbox_dict[image] = [bbox]
	


for image in image_to_bbox_dict:
	image_path = root_dir + image + '.jpg'

	image_array = io.imread(image_path)
	image_array = np.array(image_array)

	fig = plt.figure()
	ax = plt.subplot(1, 1, 1)
	ax.imshow(image_array)

	bboxes = image_to_bbox_dict[image]
	n_bboxes = len(bboxes)
	for i in range(n_bboxes):	
		bbox = bboxes[i]
		# bbox = bbox.reshape(-1)
		
		bbox[0] = float(bbox[0])
		bbox[1] = float(bbox[1])
		bbox[2] = float(bbox[2])
		bbox[3] = float(bbox[3])

		rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)

	plt.savefig(image + '_result.png', format='png')