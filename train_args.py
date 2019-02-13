import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='This is script for training face verification models.')

    parser.add_argument('--model', nargs="?", type=str, default='ChopraNet', help='Name of model')
    parser.add_argument('--loss', nargs="?", type=str, default='ContrastiveLoss', help='Name of loss function')
    parser.add_argument('--metric', nargs="?", type=int, default='1', help='Norm of distance metric')
    parser.add_argument('--lr', nargs="?", type=float, default='0.005', help='Learning rate')
    parser.add_argument('--con_epoch', nargs="?", type=int, default=0, help='Resume training')
    parser.add_argument('--exp_name', nargs="?", type=str, default="exp_1", help='Experiment name')
    args = parser.parse_args()
    print(args)
    return args
