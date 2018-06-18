import csv

# dataset = 'german'
dataset = 'belgian'

if dataset == 'german':
	csv_path = '../data/detection_data/german_test_annotations_unprocessed.txt'

if dataset == 'belgian':
	csv_path = '../data/detection_data/belgian_test_annotations_unprocessed.txt'

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


with open(dataset + '_test_annotations.txt', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)

	for key in path_to_box_dict:
		value = path_to_box_dict[key]
		row = [key]
		row.extend(value)
		spamwriter.writerow(row)