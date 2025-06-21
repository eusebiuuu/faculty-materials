import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

train_labels_all = np.load('./data/train_labels.npy')

idx = {}
cnt = 0

def load_data_train(filename):
    global cnt
    dataset = []
    with open('./data/' + filename) as f:
        dataset = f.readlines()
        for line in dataset:
            for ch in line:
                if not idx.get(ch, None):
                    idx[ch] = cnt
                    cnt += 1
    return dataset


def load_data(filename):
    dataset = []
    with open('./data/' + filename) as f:
        dataset = f.readlines()
    return dataset


train_sen = load_data_train('train_sentences.txt')
test_sen = load_data_train('test_sentences.txt')
mappings = load_data('mapping.txt')
words = load_data('words.txt')

print(len(train_sen), len(test_sen), len(train_labels_all), len(mappings), len(words))

print(train_labels_all)

# print(cnt)

def get_freq_dataset(dataset):
    real_data = []
    for line in dataset:
        bag_of_words = np.zeros((cnt))
        for ch in line:
            bag_of_words[idx[ch]] += 1
        real_data.append(bag_of_words)
    return real_data

real_train = np.array(get_freq_dataset(train_sen))
real_test = np.array(get_freq_dataset(test_sen))

# print(real_train.shape, real_test.shape)


# def normalize_data(train_data, test_data):
#     scaler = StandardScaler()
#     scaler.fit(train_data)
#     standard_train = scaler.transform(train_data)
#     standard_test = scaler.transform(test_data)
#     return (standard_train, standard_test)


def train_model(alpha):
    mse_scores = []
    mae_scores = []
    folder = KFold(n_splits=3)
    folder.get_n_splits(real_train)
    model = RidgeClassifier(alpha)

    for train_index, test_index in folder.split(real_train):
        train_data = real_train[train_index]
        test_data = real_train[test_index]
        # train_data, test_data = normalize_data(train_data, test_data)

        train_labels = train_labels_all[train_index]
        test_labels = train_labels_all[test_index]

        model.fit(train_data, train_labels)

        predicted_labels = model.predict(test_data)

        # score = model.score(test_data, test_labels)
        # print(score)

        mse_scores.append(np.mean((predicted_labels - test_labels) ** 2))
        mae_scores.append(np.mean(np.abs(predicted_labels - test_labels)))

    print(f"For alpha {alpha}:")
    print(np.mean(mse_scores))
    print(np.mean(mae_scores))

    return model


# for alpha in [1, 10, 100, 1000]:
#     train_model(alpha)

model = train_model(10)

with open('Rimboi_Eusebiu_232_subiect2_solutia_1.npy', 'w') as f:
    f.write(str(model.predict(real_test)))





