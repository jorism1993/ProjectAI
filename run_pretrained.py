from src.data_management.BelgianDataManager import BelgianDataManager
from BasicCNN.Network import *
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

device = torch.device("cpu")

belgian_data_path = './data/Belgian'
DataManager = BelgianDataManager(belgian_data_path)
DataLoader = DataManager.load_data(resize = (64,64), batch_size=16)
TrainDataLoader = DataLoader['train']
TestDataLoader = DataLoader['test']


num_classes = len(TrainDataLoader.dataset.classes)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, TrainDataLoader, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, TrainDataLoader, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)