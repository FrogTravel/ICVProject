import cv2
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
        sample = self._pad_if_needed(sample)
        image, landmarks, bbox = sample['image'], sample['landmarks_2d'], sample['bbox']
        h, w = image.shape[:2]
        image = self.transform(image=image)['image']
        new_h, new_w = image.shape[:2]
        landmarks = landmarks * torch.tensor([new_w / w, new_h / h])
        bbox = [bbox[0] * new_w / w, bbox[1] * new_h / h, bbox[2] * new_w / w,
                bbox[3] * new_h / h]
        return {'image': np.array(image), 'landmarks_2d': landmarks, 'bbox': torch.tensor(bbox)}

    def _pad_if_needed(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks_2d'], sample['bbox']
        h, w = image.shape[:2]
        top, bottom, left, right = self._get_borders_size(landmarks, h, w)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        bbox = [bbox[0] + left, bbox[1] + top, bbox[2], bbox[3]]
        landmarks = landmarks + torch.tensor([[left, top]] * 68).float()
        return {'image': np.array(image), 'landmarks_2d': landmarks, 'bbox': bbox}

    def _get_borders_size(self, landmarks, h, w):
        top, bottom, left, right = 0.0, 0.0, 0.0, 0.0
        for i, xy in enumerate(landmarks):
            x = landmarks[i][0]
            y = landmarks[i][1]
            if x <= 0:
                left = max(abs(x - 1.0), left)
            elif landmarks[i][0] >= w:
                right = max(x + 1.0, right)

            if y <= 0:
                top = max(abs(y - 1.0), top)
            elif y >= h:
                bottom = max(y + 1.0, bottom)
        return int(top), int(bottom), int(left), int(right)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        max_cut_fraction (tuple or float): Desired fraction of. If int, square crop
            is made.
    """

    def __init__(self, max_cut_fraction):
        assert isinstance(max_cut_fraction, (float, tuple))
        if isinstance(max_cut_fraction, float):
            assert 0 <= max_cut_fraction < 1
            self.max_cut_fraction = (max_cut_fraction, max_cut_fraction)
        else:
            assert len(max_cut_fraction) == 2
            self.max_cut_fraction = max_cut_fraction

    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks_2d'], sample['bbox']

        h, w = image.shape[:2]
        h_max_offset, w_max_offset = self.max_cut_fraction
        top = np.random.randint(0, h_max_offset * h)
        left = np.random.randint(0, w_max_offset * w)

        image = image[top: top + h,
                left: left + w]

        landmarks = landmarks - torch.tensor([left, top]).float()

        bbox = [bbox[0] - left, bbox[1] - top, bbox[2],
                bbox[3]]

        return {'image': image, 'landmarks_2d': landmarks, 'bbox': torch.tensor(bbox)}
