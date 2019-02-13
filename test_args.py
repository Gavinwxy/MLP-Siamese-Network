import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='This is script for testing face verification models.')

    parser.add_argument('--exp_name', nargs="?", type=str, default="exp_1", help='Experiment name')
    args = parser.parse_args()
    print(args)
    return args
