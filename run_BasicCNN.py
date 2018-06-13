from src.data_management.BelgianDataManager import BelgianDataManager
from src.data_management.GermanDataManager import GermanDataManager
# from BasicCNN.Network import *
from BasicCNN.STN import *
import torch.optim as optim

from collections import Counter



belgian_data_path = './data/Belgian'
german_data_path = './data/German'
# DataManager = BelgianDataManager(belgian_data_path)
DataManager = GermanDataManager(german_data_path)
DataLoader = DataManager.load_data(resize = (28,28), batch_size=64)
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


# CNN = Net(num_classes, input_size=16)




# print(len(TestDataLoader))
# print('Start training the model')
# train(CNN, TrainDataLoader, TestDataLoader, optimizer, number_of_epochs=25)
# print('Perform a test')
# test(CNN, TestDataLoader)

#Run STN

STN = Net(num_classes)

optimizer = optim.Adam(STN.parameters(), lr=0.001)

number_of_epochs = 25

for epoch in range(1, number_of_epochs):
    train(epoch, TrainDataLoader, STN, optimizer)
    test(TestDataLoader, STN)