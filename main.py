import torch
import argparse
from solution import Solution
from model import EEGNet, DepthwiseSeparableConv2d

def main(args):
    solution = Solution(args)


    if args.mode == "train":
        solution.train()
    elif args.mode == "test":
        solution.test()
    elif args.mode == "attack":
        solution.attack(method=args.attack_method, target=args.target, epsilon=args.epsilon, iteration=args.iteration)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SSVEP-FGSM")
    parser.add_argument('--epoch', type=int, default=150, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate of optimizer")
    parser.add_argument('--nb_classes', type=int, default=40, help="number of classes")
    parser.add_argument('--channels', type=int, default=9, help="number of eeg channels")
    parser.add_argument('--samp_rate', type=int, default=250, help="number of sample rate")

    parser.add_argument('--attack_method', type=str, default="CW", help='FGSM / iFGSM / CW / PGD')
    parser.add_argument('--iteration', type=int, default=-1, help='the number of iteration for FGSM')
    parser.add_argument('--target', type=int, default=15, help='target class for targeted generation')
    parser.add_argument('--epsilon', type=float, default=0.001, help='epsilon for FGSM and i-FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM Attack')
    parser.add_argument('--initial_const', type=float, default=1.0, help='initial_const of CW Attack')
    parser.add_argument('--cuda', type=bool, default=True, help='enable cuda')

    parser.add_argument('--dataset', type=str, default='Benchmark', help='dataset type')
    parser.add_argument('--env_name', type=str, default='pytorch', help='environment name')
    parser.add_argument('--mode', type=str, default='attack', help='train / test / attack')
    parser.add_argument('--model', type=str, default='EEGNet', help='EEGnet / DeepCNN / ShallowCNN')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--checkpoint', type=str, default='best_acc2.tar', help="checkpoint's file name")

    # parser.add_argument('--train', type=str, default=True, help="Train or False")
    parser.add_argument('--samples', type=int, default=1375, help="length of eeg data samples")

    opt = parser.parse_args()

    main(opt)
