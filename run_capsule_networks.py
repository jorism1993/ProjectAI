from src.data_management.BelgianDataManager import BelgianDataManager
from capsule_networks.capsule_network import train

belgian_data_path = './data/Belgian'
DataManager = BelgianDataManager(belgian_data_path)
DataLoader = DataManager.load_data(resize = (64,64))
TrainDataLoader = DataLoader['train']
TestDataLoader = DataLoader['test']

num_classes = len(TrainDataLoader.dataset.classes)

print('Start training the model')
train(dataloader=TrainDataLoader, BATCH_SIZE=100, NUM_CLASSES=num_classes,
            NUM_EPOCHS=500, NUM_ROUTING_ITERATIONS=3)