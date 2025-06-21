import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    dataset = []
    with open('./data/' + filename) as f:
        dataset = f.readlines()
    return dataset

train_sen = load_data('train_sentences.txt')
test_sen = load_data('test_sentences.txt')
mappings = load_data('mapping.txt')
words = load_data('words.txt')

idx = {}

for i, mapping in enumerate(mappings):
    if i != 9:
        # print(mapping)
        ch, num = mapping.split(',')
        idx[ch] = int(num)
    else:
        idx[','] = 10


def convert_string_to_array(string):
    word_arr = []
    for ch in string:
        word_arr.append(idx.get(ch, 0))
    return word_arr

def convert_list_to_arr(curr_list):
    res = []
    for elem in curr_list:
        res.append(convert_string_to_array(elem))
    return res

new_words = convert_list_to_arr(words)
new_test_sen = convert_list_to_arr(test_sen)
new_train_sen = convert_list_to_arr(train_sen)

def get_norm(arr1, arr2):
    sum = 0
    for i in range(len(arr1)):
        sum += (arr1[i] - arr2[i]) ** 2
    return np.sqrt(sum)


def convolution(data_arr, n_gram, dim = 3):
    res = []
    for i in range(len(data_arr) - dim + 1):
        prod = 0
        for j in range(0, dim):
            prod += data_arr[i + j] * n_gram[j]
        norm = get_norm(data_arr[i:i + dim], n_gram)
        prod /= norm
        res.append(prod)
    
    res[res >= 0.9] = 1
    
    cnt = 0
    for elem in res:
        cnt += elem == 1

    return res




