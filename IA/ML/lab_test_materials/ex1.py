from sklearn.svm import SVC
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def get_frequency_vector(image, d:int):
    patch_row = (d - 1) // 2
    patch_column = (d - 1) // 2
    rows = len(image)
    cols = len(image[0])

    padded_image = np.pad(image, pad_width=((patch_row, d - patch_row), (patch_column, d - patch_column)), 
                          constant_values=(-1, -1))

    freq_array = np.zeros(1 << (d ** 2 - 1))

    for r in range(patch_row, rows):
        for c in range(patch_column, cols):
            focus_image = padded_image[r - patch_row:r + (d - patch_row), c - patch_column:c + (d - patch_column)]

            neighbours = []
            for i in range(0, d):
                for j in range(0, d):
                    if i != patch_row or j != patch_column:
                        neighbours.append(focus_image[i][j])
            
            comparison_result = padded_image[r][c] < neighbours
            converted_num = 0
            
            for i in range(0, len(comparison_result)):
                converted_num += (1 << i) * int(comparison_result[i])
            
            freq_array[converted_num] += 1
    
    return freq_array


def load():
    with open("data.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

train_images, train_labels, test_images, test_labels = load()

train_images = train_images.reshape(-1, 28, 28)
test_images = test_images.reshape(-1, 28, 28)

def process_data(d):
    train_data = [get_frequency_vector(image, d) for image in tqdm(train_images)]
    test_data = [get_frequency_vector(image, d) for image in tqdm(test_images)]
    return train_data, test_data


scaler = StandardScaler()
train_data, test_data = process_data(3)
train_data = np.array(train_data)
test_data = np.array(test_data)

print(train_data.shape, test_data.shape)

scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

model = SVC(C = 1, kernel='rbf', gamma=0.001)
model.fit(scaled_train, train_labels)

print("Accuracy:", model.score(scaled_test, test_labels) * 100, "%")
