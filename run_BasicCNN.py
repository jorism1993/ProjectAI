from src.data_management.BelgianDataManager import BelgianDataManager
from BasicCNN.Network import *
import torch.optim as optim

from collections import Counter



belgian_data_path = './data/Belgian'
DataManager = BelgianDataManager(belgian_data_path)
DataLoader = DataManager.load_data(resize = (32,32), batch_size=1)
TrainDataLoader = DataLoader['train']
TestDataLoader = DataLoader['test']


def print_statistics():
    classes = [int(label.data[0]) for input, label in TrainDataLoader]
    count_classes_train = Counter(classes)
    print('Overview Training Set')
    for k in count_classes_train.keys():
        print('Class {} has {} occurences'.format(k, count_classes_train[k]))

    classes = [int(label.data[0]) for input, label in TestDataLoader]
    count_classes = Counter(classes)

    print('Overview Test Set')
    for k in count_classes.keys():
        print('Class {} has {} occurences'.format(k, count_classes[k]))

    print('The trainingsset contains {} items whereas the test set only contains {} items'.
            format(sum(count_classes_train.values()), sum(count_classes.values())))

num_classes = len(TrainDataLoader.dataset.classes)


CNN = Net(num_classes, input_size=16)


optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)

print(len(TestDataLoader))
print('Start training the model')
train(CNN, TrainDataLoader, TestDataLoader, optimizer, number_of_epochs=3)
print('Perform a test')
test(CNN, TestDataLoader)