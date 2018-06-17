from data_loader import DetectionDataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image

root_folder = '../data/test_data/'

data_transform = transforms.Compose([
    ])

test_dataset = DetectionDataset(root_dir=root_folder, transform=data_transform, use_data='test', use_superclass=True)

for i in range(len(test_dataset)):
	data = test_dataset[i]
	image = data[0]
	image_info = data[1]
	bboxes = data[2][0]
	n_bboxes = data[3]

	image = np.squeeze(image, axis=0)
	image = np.moveaxis(image, 0, -1)
	img_pil = Image.fromarray(image)

	for j in range(n_bboxes):	
		bbox = bboxes[j]
		bbox = bbox.reshape(-1)
		
		bbox[0] = bbox[0] * image_info[1]
		bbox[1] = bbox[1] * image_info[0]
		bbox[2] = bbox[2] * image_info[1]
		bbox[3] = bbox[3] * image_info[0]

		img_path = root_folder + 'cropped_video_images/' + str(i) + '_' + str(j) + '.jpg'
		try:
			img_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3])).save(img_path)
		except Exception as e:
			print('ERROR: invalid bounding box')
		
