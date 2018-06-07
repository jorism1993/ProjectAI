import csv

csv_path = '../data/detection_data/annotations.txt'

path_to_box_dict = {}

with open(csv_path, newline='') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')

	for row in csv_reader:
		file_name = row[0]
		info = row[1:7]

		if file_name in path_to_box_dict:
			path_to_box_dict[file_name].extend(info)
		else:
			path_to_box_dict[file_name] = info


with open('annotations_multiple_boxes.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)

	for key in path_to_box_dict:
		value = path_to_box_dict[key]
		row = [key]
		row.extend(value)
		spamwriter.writerow(row)