import csv
from lxml import etree
from PIL import Image
import os.path

datasets = ['belgian', 'german']
if len(datasets) == 2:
	folder_name = 'results/both/Annotations'
else:
	folder_name = 'results/{}/Annotations'.format(datasets[0])
# dataset = 'belgian'

for dataset in datasets:
	if dataset == 'german':
		csv_path = 'data/detection_data/german_annotations.txt'
		bbox_length = 7

	if dataset == 'belgian':
		csv_path = 'data/detection_data/belgian_annotations.txt'
		bbox_length = 6

	path_to_box_dict = {}

	with open(csv_path, newline='') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')

		for row in csv_reader:
			file_name = row[0]
			info = row[1:7]

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

	for file in path_to_box_dict:
		# Get appropiate image
		image_path = csv_path.split('/')[:-1]
		if dataset == 'german':
			image_path.append('german_resized')
		if dataset == 'belgian':
			image_path.append('belgian_resized')
		image_path.append(file)
		image_path = '/'.join(image_path)
		pil_image = Image.open(image_path)
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

			bnbbox = etree.SubElement(object, 'bnbbox')
			xmin = etree.SubElement(bnbbox, 'xmin')
			xmin.text = bbox[0]
			ymin = etree.SubElement(bnbbox, 'ymin')
			ymin.text = bbox[1]
			xmax = etree.SubElement(bnbbox, 'xmax')
			xmax.text = bbox[2]
			ymax = etree.SubElement(bnbbox, 'ymax')
			ymax.text = bbox[3]

		tree = etree.ElementTree(root)
		name = os.path.splitext(file)[0]
		path = folder_name + '/' + name + '.xml'
		tree.write(path, pretty_print=True, encoding="utf-8")
