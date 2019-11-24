import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import albumentations as albu
import torch
from torchvision import transforms


class LS3DDataset(Dataset):
    """LS3D-W - 3d Face Landmarks dataset"""

    def __init__(self, csv_file, root_dir, albu_transformations=None, transformations=None):
        """
        :param csv_file: string. Path to the csv file with annotations.
        :param root_dir: string. Directory with all the images.
        :param albu_transformations: list or albumentation transform. Optional transform to be applied
                on a sample.
        :param transformations: list of torch or other transform
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = None
        self.albu_transformer = None
        if albu_transformations:
            self.albu_transformer = self.create_transformer(albu_transformations)
        if transformations:
            self.transform = transforms.Compose(transformations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: index of an image
        :return:
        {'image': tensor, 'bbox': torch.size(num_of_faces, 4) [center_x, center_y, width, height],
        'landmarks_2d': torch.size(num_of_faces,68, 2)}
        num_of_faces = 1
        """
        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx]['file_name'])
        image = np.array(Image.open(img_name))

        landmarks_2d = torch.tensor(self.data.iloc[idx][[col for col in self.data.columns if col.startswith('2d')]],
                                    dtype=torch.float)
        landmarks_2d = landmarks_2d.view(-1, 2)

        sample = {'image': image, 'landmarks_2d': landmarks_2d,
                  'bbox': self.get_bbox(landmarks_2d)}

        if self.transform:
            sample = self.transform(sample)

        if self.albu_transformer:
            return self.apply_albu_transform(sample)

        return {'image': torch.tensor(sample['image']),
                'landmarks_2d': sample['landmarks_2d'].unsqueeze(0),
                'bbox': self.get_formatted_bbox(sample['bbox'], sample['image'].shape)}

    def create_transformer(self, transformations):
        """
        'coco' format for bbox: [x_min, y_min, width, height]
        :param transformations: list of transformations
        :return: callable transform
        """
        return albu.Compose(transformations,
                            keypoint_params={'format': 'xy'},
                            bbox_params={'format': 'coco', 'label_fields': ['category_id']})

    def apply_albu_transform(self, sample):
        landmarks = [(xy[0], xy[1]) for xy in sample['landmarks_2d']]
        transformed_sample = self.albu_transformer(
            image=sample['image'], keypoints=landmarks, bboxes=sample['bbox'].unsqueeze(0),
            category_id=[1])
        landmarks = torch.tensor([[xy[0], xy[1]] for xy in transformed_sample['keypoints']]).unsqueeze(0)
        bbox = self.get_formatted_bbox(torch.tensor(transformed_sample['bboxes']), transformed_sample['image'].shape)
        transformed_sample = {'image': transformed_sample['image'], 'landmarks_2d': landmarks, 'bbox': bbox}
        return transformed_sample

    def get_bbox(self, landmarks):
        """
        Finds bounding box coordinates in format [xmin, ymin, width, height] from landmarks
        :param landmarks: torch.size(68,2)
        :return: torch.size(4)
        """

        x = landmarks[:, 0]
        y = landmarks[:, 1]

        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        width = xmax - xmin
        height = ymax - ymin
        return torch.tensor([xmin, ymin, width, height])

    def get_formatted_bbox(self, bbox, image_size):
        """

        :param bbox: [[xmin, ymin, width, height]]
        :param image_size: (tuple): (height, width)
        :return: [[center_x, center_y, w, h]]
        where w, h are width and height of the bbox in [0,1] w.r.t. the whole image
        """
        xmin, ymin, width, height = bbox[0]
        c_x = (xmin + width / 2) / image_size[1]
        c_y = (ymin + height / 2) / image_size[0]
        height /= image_size[0]
        width /= image_size[1]

        return torch.tensor([c_x, c_y, width, height]).unsqueeze(0)
