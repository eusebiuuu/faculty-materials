import os
import numpy as np

directory = './data/'
MIN_SIZE = 145

def find_min_size(filename):
    min_size = 150

    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue

            file_name, _ = line.split(',')

            with open(directory + file_name) as g:
                min_size = min(min_size, len(g.readlines()))
            
    print("Min size: " + str(min_size))


def load_files(filename):
    dataset = []
    labels = []

    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue

            file_name, label = line.split(',')
            labels.append(label)
            
            curr_data = []
            with open(directory + file_name) as g:
                for curr_line in g.readlines():
                    vals = curr_line.split(',')
                    curr_data.append([float(x) for x in vals])

            dataset.append(curr_data[0:MIN_SIZE])

    return dataset, labels
    

# find_min_size('train.txt')
# find_min_size('test_labels.txt')

train_dataset, train_labels = load_files('train.txt')
test_dataset, test_labels = load_files('test_labels.txt')
print(len(train_dataset), len(test_dataset))