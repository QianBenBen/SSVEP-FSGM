import torch
import argparse
from train import eeg_train
from test import eeg_test
from model import EEGNet, DepthwiseSeparableConv2d

parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, default=True, help="Train or False" )
parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate of optimizer")
parser.add_argument('--nb_classes', type=int, default=40, help="number of classes")
parser.add_argument('--channels', type=int, default=9, help="number of eeg channels" )
parser.add_argument('--samples', type=int, default=1375, help="number of eeg data samples")
parser.add_argument('--sample_rate', type=int, default=250, help="number of sample rate")

opt  = parser.parse_args()




if __name__ == '__main__':
    if opt.train == True:
        eeg_train(learning_rate=opt.learning_rate, nb_classes=opt.nb_classes, Chans=opt.channels, Samples=opt.samples, num_epochs=200)
    elif opt.train == False:
        eeg_test()

