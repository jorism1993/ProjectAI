import csv

csv_path = '../data/detection_data/annotations.txt'

path_to_box_dict = {}

with open(csv_path, newline='') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')

	for row in csv_reader:
		file_name = row[0]
		info = row[1:7]

		if info[-1] == '-1':
			info[-1] = '11'
			
		if info[-2] == '-1':
			info[-2] = '11'

		# if file_name[0:2] == '04':
		if file_name in path_to_box_dict:
			path_to_box_dict[file_name].extend(info)
		else:
			path_to_box_dict[file_name] = info


with open('annotations_multiple_boxes_only4.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)

	for key in path_to_box_dict:
		value = path_to_box_dict[key]
		row = [key]
		row.extend(value)
		spamwriter.writerow(row)