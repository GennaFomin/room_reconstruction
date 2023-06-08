import numpy as np
from utils import load_image_url, autocrop, image_load_path
from model import fit_svr
from patches import extract_patches

import fire

def score_pipline_url(image):
    score_list = []

    image = load_image_url(image)
    image = autocrop(image)
    image_patches = extract_patches(image, (1000, 1000))

    for i in image_patches:
        score_list.append(fit_svr(i))


    return print((np.percentile(score_list, 70), '- percentile 70%'))

def score_pipline_path(image):
    score_list = []

    image = image_load_path(image)
    image = autocrop(image)
    image_patches = extract_patches(image, (1000, 1000))

    for i in image_patches:
        score_list.append(fit_svr(i))


    return print((np.percentile(score_list, 70), '- percentile 70%'))

if __name__ == '__main__':
    fire.Fire()