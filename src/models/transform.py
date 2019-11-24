import torch
import numpy as np
import albumentations as albu


class Rescale(object):

    def __init__(self, output_size):
        """
        Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
        :param output_size: (int) maximum size of the image after the transformation
        """
        self.transform = albu.LongestMaxSize(output_size)

    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks_2d'], sample['bbox']
        h, w = image.shape[:2]

        image = self.transform(image=image)['image']
        new_h, new_w = image.shape[:2]
        landmarks = landmarks * torch.tensor([new_w / w, new_h / h])
        bbox = [bbox[0] * new_w / w, bbox[1] * new_h / h, bbox[2] * new_w / w, bbox[3] * new_h / h]

        return {'image': np.array(image), 'landmarks_2d': landmarks, 'bbox': torch.tensor(bbox)}
