""" File to create training and testing data sets out the
    selected HPCs.
    Author: Zander Blasingame """

import argparse
from random import shuffle

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--p_train',
                    type=float,
                    default=0.5,
                    help=('Float between 0 and 1 that describes the '
                          'percentage of training data to be pulled '
                          'from the dataset'))
parser.add_argument('--train_path',
                    type=str,
                    default='',
                    help='Location of output train file')
parser.add_argument('--test_path',
                    type=str,
                    default='',
                    help='Location of output test file')
parser.add_argument('--input_file',
                    type=str,
                    default='data.csv',
                    help='Location of input file')
parser.add_argument('--random',
                    action='store_true',
                    help='Flag to shuffle data')

args = parser.parse_args()

lines = []
header = []
is_header = True

with open(args.input_file, 'r') as f:
    for line in f:
        if line[0] == '#':
            continue

        if is_header:
            header = line
            is_header = False
            continue

        lines.append(line)

# Shuffle and split code into testing and training
shuffle(lines)

point = int(len(lines) * args.p_train)

train_lines = [header] + lines[:point]
test_lines = [header] + lines[point:]

# write to files
with open(args.train_path + 'train.csv', 'w') as f:
    f.writelines(train_lines)

with open(args.test_path + 'test.csv', 'w') as f:
    f.writelines(test_lines)
