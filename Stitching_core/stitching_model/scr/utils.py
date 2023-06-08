import skimage.io
import numpy as np
import cv2
import urllib.request


def load_image_url(url):
    image_stream = urllib.request.urlopen(url)
    image = skimage.io.imread(image_stream, plugin='pil')

    return skimage.color.rgb2gray(image)


def image_load_path(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
