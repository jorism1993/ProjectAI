import csv


csv_path = '../data/test_data/video_annotation.csv'

path_to_box_dict = {}

with open(csv_path, newline='') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

	for row in csv_reader:
		if row[0] == 'filename':
			continue

		file_name = row[1]
		class_id = row[2]

		xmin = row[4]
		ymin = row[3]
		xmax = row[6]
		ymax = row[5]

		info = [xmin, ymin, xmax, ymax, class_id, class_id]

		# if file_name[0:2] == '04':
		if file_name in path_to_box_dict:
			path_to_box_dict[file_name].extend(info)
		else:
			path_to_box_dict[file_name] = info


with open('video_new_annotations.txt', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)

	for key in path_to_box_dict:
		value = path_to_box_dict[key]
		row = [key]
		row.extend(value)
		spamwriter.writerow(row)