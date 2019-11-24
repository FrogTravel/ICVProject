from PIL import Image
from matplotlib import pyplot as plt, patches
import numpy as np


def draw_image_with_landmarks_bbox(image, landmarks, bbox):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1])
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def draw_image_with_landmarks(image, landmarks):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()


def read_image(image_path):
    return np.array(Image.open(image_path))


import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
input = io.imread('/root/data/LS3D-W/300W-Testset-3D/outdoor_177.png')
preds = fa.get_landmarks(input)
draw_image_with_landmarks(read_image('/root/data/LS3D-W/300W-Testset-3D/outdoor_177.png'), preds[0])
