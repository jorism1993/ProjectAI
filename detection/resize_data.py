from data_loader import DetectionDataset
import numpy as np
from PIL import Image
import csv


root_folder = '../faster_rcnn/data/detection_data/'

german_dataset = DetectionDataset(root_dir=root_folder, use_data='belgian', use_superclass=True)
csv_array = []

for i in range(len(german_dataset)):
	data = german_dataset[i]
	if data[3] == None:
		continue
	
	image = data[0]
	bboxes = data[2]
	n_bboxes = data[3]
	bboxes = bboxes[:, 0:n_bboxes, :]
	bboxes = bboxes[0]

	superclass_column = bboxes[:, -1]
	superclass_column = np.reshape(superclass_column, (-1, 1))
	bboxes = np.hstack((bboxes, superclass_column))
	bboxes = bboxes.flatten()

	image_path = german_dataset.get_file_path(i)
	image_path = image_path.split('/')[-1]

	row = [image_path]
	row.extend(bboxes)

	image = image[0]
	image = np.moveaxis(image, 0, -1)
	
	im = Image.fromarray(image)
	im.save(root_folder + 'belgian_resized/' + image_path)
	csv_array.append(row)


with open(root_folder + 'belgian_resized_annotations.txt', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)

	for row in csv_array:
		spamwriter.writerow(row)