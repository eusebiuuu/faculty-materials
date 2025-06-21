import numpy as np
import matplotlib.pyplot as plt

images = []

for i in range(0, 9):
    curr_image = np.load(f"./images/car_{i}.npy")
    images.append(curr_image)

print(np.sum(images))

sums = [int(np.sum(image)) for image in images]
print(sums)

print(np.argmax(sums))

mean_image = np.mean(images, axis=0)
plt.imshow(mean_image)
# plt.show()

std_dev = np.std(images)
print(std_dev)

normalized_images = [np.divide(np.subtract(image, mean_image), std_dev) for image in images]

plt.imshow(normalized_images[5][200:301,280:401])
plt.show()