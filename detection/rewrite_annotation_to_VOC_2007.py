import csv
from lxml import etree
from PIL import Image
import os.path

train_split = 0.7
test_split = 0.15
trainval_split = 0.075
val_split = 0.075


datasets = ['german']
if len(datasets) == 2:
	folder_name = 'results/both/'
else:
	folder_name = 'results/{}/'.format(datasets[0])
# dataset = 'belgian'

for dataset in datasets:
	if dataset == 'german':
		csv_path = '../data/detection_data/german_jpg_annotations.txt'
		bbox_length = 7

	if dataset == 'belgian':
		csv_path = '../data/detection_data/belgian_resized_annotations.txt'
		bbox_length = 6

	path_to_box_dict = {}

	with open(csv_path, newline='') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')

		for row in csv_reader:
			file_name = row[0]

			n_bboxes = int((len(row) - 1) / 6)
			for i in range(n_bboxes):

				info = row[(i*6+1):(i*6+7)]

				if dataset == 'german':
					info.append(info[-1])

				if info[-1] == '-1':
					info[-1] = '11'

				if info[-2] == '-1':
					info[-2] = '11'

				# if file_name[0:2] == '04':
				if file_name in path_to_box_dict:
					path_to_box_dict[file_name].extend(info)
				else:
					path_to_box_dict[file_name] = info


	for index, file in enumerate(path_to_box_dict):


		# Get appropiate image
		image_path = csv_path.split('/')[:-1]
		if dataset == 'german':
			image_path.append('german_jpg')
		if dataset == 'belgian':
			image_path.append('belgian_resized')

		image_path.append(file.split('/')[-1])

		if dataset == 'belgian':
			image_path[-1] = image_path[-1][:-3] + 'jpg'



		image_path = '/'.join(image_path)
		try:
			pil_image = Image.open(image_path)
		except Exception as e:
			print('image not found:', image_path)
			# print('image ' +  str(image_path), + ' not found')
			continue

		image_width, image_height = pil_image.size

		# Create XML tree
		root = etree.Element("annotation")

		folder = etree.SubElement(root, 'folder')
		folder.text = 'VOC2007'

		filename = etree.SubElement(root, 'filename')
		filename.text = file

		source = etree.SubElement(root, 'source')
		database = etree.SubElement(source, 'database')
		database.text = 'The VOC2007 Database'
		annotation = etree.SubElement(source, 'annotation')
		annotation.text = 'PASCAL VOC2007'
		image = etree.SubElement(source, 'image')
		image.text = 'flickr'
		flickrid = etree.SubElement(source, 'flickrid')
		flickrid.text = '0'

		owner = etree.SubElement(root, 'owner')
		flickrid = etree.SubElement(owner, 'flickrid')
		flickrid.text = 'Group 27'
		name = etree.SubElement(owner, 'name')
		name.text = '?'

		size = etree.SubElement(root, 'size')
		width = etree.SubElement(size, 'width')
		width.text = str(image_width)
		height = etree.SubElement(size, 'height')
		height.text = str(image_height)
		depth = etree.SubElement(size, 'depth')
		depth.text = '3'

		segmented = etree.SubElement(root, 'segmented')
		segmented.text = '0'

		# Add bounding boxes to XML
		n_bboxes = len(path_to_box_dict[file])
		
		n_bboxes /= bbox_length
		for i in range(int(n_bboxes)):
			bbox = path_to_box_dict[file][i * bbox_length: i * bbox_length + bbox_length - 1]

			object = etree.SubElement(root, 'object')

			name = etree.SubElement(object, 'name')
			name.text = 'traffic_sign'
			pose = etree.SubElement(object, 'pose')
			pose.text = 'Unspecified'
			truncated = etree.SubElement(object, 'truncated')
			truncated.text = '0'
			difficult = etree.SubElement(object, 'difficult')
			difficult.text = '0'

			bnbbox = etree.SubElement(object, 'bndbox')
			xmin = etree.SubElement(bnbbox, 'xmin')
			xmin.text = bbox[0]
			ymin = etree.SubElement(bnbbox, 'ymin')
			ymin.text = bbox[1]
			xmax = etree.SubElement(bnbbox, 'xmax')
			xmax.text = bbox[2]
			ymax = etree.SubElement(bnbbox, 'ymax')
			ymax.text = bbox[3]

		tree = etree.ElementTree(root)
		if dataset == 'belgian':
			file = file.split('/')[-1]
		name = os.path.splitext(file)[0]
		path = folder_name + 'Annotations/' + name + '.xml'



		if (index/len(path_to_box_dict)) < train_split:
			with open(folder_name + 'train.txt', 'a') as myfile:
				myfile.write(name + '\n')
			with open(folder_name + 'traffic_sign_train.txt', 'a') as myfile:
				myfile.write(name + ' -1\n')
		elif (index/len(path_to_box_dict)) < (train_split + test_split):
			with open(folder_name + 'test.txt', 'a') as myfile:
				myfile.write(name + '\n')
			with open(folder_name + 'traffic_sign_test.txt', 'a') as myfile:
				myfile.write(name + ' -1\n')
		elif (index/len(path_to_box_dict)) < (train_split + test_split + trainval_split):
			with open(folder_name + 'trainval.txt', 'a') as myfile:
				myfile.write(name + '\n')
			with open(folder_name + 'traffic_sign_trainval.txt', 'a') as myfile:
				myfile.write(name + ' -1\n')
		elif (index/len(path_to_box_dict)) < (train_split + test_split + trainval_split + val_split):
			with open(folder_name + 'val.txt', 'a') as myfile:
				myfile.write(name + '\n')
			with open(folder_name + 'traffic_sign_val.txt', 'a') as myfile:
				myfile.write(name + ' -1\n')

		tree.write(path, pretty_print=True, encoding="utf-8")

		
