import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

train_images = np.loadtxt('./data/train_images.txt', converters=float)
train_labels = np.loadtxt('./data/train_labels.txt', 'int', converters=float)
test_images = np.loadtxt('./data/test_images.txt', converters=float)
test_labels = np.loadtxt('./data/test_labels.txt', 'int', converters=float)

def get_score(bins_count):
    color_bins = np.linspace(0, 255, bins_count, endpoint=False)

    def discretize_image(image, bins):
        new_image = []
        for sample in image:
            new_image.append(np.digitize(sample, bins) - 1)
        return new_image

    processed_test_images = []
    processed_train_images = []

    for image in train_images:
        image = np.reshape(image, (28, 28))
        processed_train_images.append(discretize_image(image, color_bins))

    for image in test_images:
        image = np.reshape(image, (28, 28))
        processed_test_images.append(discretize_image(image, color_bins))

    processed_test_images = np.array(processed_test_images)
    processed_train_images = np.array(processed_train_images)

    # sample_image = processed_train_images[0, :]
    # plt.imshow(sample_image.astype(np.uint8), cmap='gray')
    # plt.show()

    model = MultinomialNB()

    processed_train_images = processed_train_images.reshape(-1, 28 * 28)
    processed_test_images = processed_test_images.reshape(-1, 28 * 28)
    model.fit(processed_train_images, train_labels)

    predicted_labels = model.predict(processed_test_images)

    # wrong_count = 0
    # for (i, label) in enumerate(predicted_labels):
    #     if label != test_labels[i]:
    #         print(label, test_labels[i])
    #         plt.imshow(np.array(processed_test_images[i]).reshape(28, 28).astype(np.uint8), cmap='gray')
    #         plt.show()

    #         wrong_count += 1
    #         if wrong_count == 10:
    #             break

    confusion_matrix = np.zeros((10, 10))
    for (i, label) in enumerate(predicted_labels):
        confusion_matrix[test_labels[i]][label] += 1

    print(confusion_matrix)
    return model.score(processed_test_images, test_labels)

# for c in [3, 5, 7, 9, 11]:
#     print(f"{get_score(c):.5f}")

print(get_score(4))
