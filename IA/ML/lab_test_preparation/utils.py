import numpy as np

def translate_image(img, tx, ty):
    """
    Translate an image using NumPy.

    Parameters:
    - img: np.ndarray, image array (H x W x C) or (H x W)
    - tx: int, horizontal translation (positive = right, negative = left)
    - ty: int, vertical translation (positive = down, negative = up)

    Returns:
    - Translated image as np.ndarray
    """
    # Determine image shape
    h, w = img.shape[:2]

    # Create a blank canvas of the same shape
    translated = np.zeros_like(img)

    # Calculate ranges for source and destination
    src_x_start = max(-tx, 0)
    src_x_end = min(w - tx, w)
    dst_x_start = max(tx, 0)
    dst_x_end = min(w + tx, w)

    src_y_start = max(-ty, 0)
    src_y_end = min(h - ty, h)
    dst_y_start = max(ty, 0)
    dst_y_end = min(h + ty, h)

    # Copy the pixel values from source to destination
    translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img[src_y_start:src_y_end, src_x_start:src_x_end]

    return translated


from scipy.ndimage import rotate

# Rotate 45 degrees counter-clockwise
rotated = rotate(matrix, angle=45, reshape=True)

import numpy as np

matrix = np.array([[1, 2],
                   [3, 4]])

# Rotate 90 degrees counter-clockwise
rot90 = np.rot90(matrix, k=1)

# Rotate 180 degrees
rot180 = np.rot90(matrix, k=2)

# Rotate 270 degrees (or 90 degrees clockwise)
rot270 = np.rot90(matrix, k=3)


import numpy as np

# Assume arr has shape (28, 28, 5000)
arr = np.random.rand(28, 28, 5000)

# Transpose the axes to (5000, 28, 28)
reshaped = np.transpose(arr, (2, 0, 1))

print(reshaped.shape)  # Output: (5000, 28, 28)