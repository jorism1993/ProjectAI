import argparse

from src.data_management.BelgianDataManager import BelgianDataManager
from ssd.pytorch.train import train, test

belgian_data_path = './data/...'
DataManager = BelgianDataManager(belgian_data_path)
DataLoader = BelgianDataManager.load_data(resize = (400,400))
TrainDataLoader = DataLoader['train']
TestDataLoader = DataLoader['test']

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--min_dim', default=0,
                    help='Minimal number of dimensions')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--epochs', default=25, type=float,
                    help='Number of epochs')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

num_classes = len(TrainDataLoader.image_datasets['train'].classes)
lr_steps = [80000, 100000, 120000]

train(data_loader=TrainDataLoader, min_dim=args.min_dim, num_classes=num_classes,
            cuda=args.cuda, basenet=args.basenet, batch_size=args.batch_size,
            checkpoint=args.resume, start_iter=args.start_iter, workers=args.num_workers,
            lr=args.lr, number_of_epochs=args.epochs, momentum=args.momentum,
            lr_steps=lr_steps, weight_decay=args.weight_decay, gamma=args.gamma,
            use_visdom=args.visdom, save_folder=args.save_folder)

test()