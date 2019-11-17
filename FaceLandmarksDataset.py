import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


def read_and_resize(filename):
    img = Image.open(filename)
    img = img.resize((300, 300))
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    img_t = transform(img)
    return img_t


def get_bbox(landmarks):
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
    return [xmin, ymin, width, height]


class FaceLandmarksDataset(Dataset):
    """Face dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_name = self.dataset['file_name'][idx]
        image = read_and_resize(image_name)
        coordinates_int = []
        for i in range(0, len(self.dataset.iloc[idx][1:])):
            if i % 2 == 0:
                (x, y) = ((int)(self.dataset.iloc[idx][1:][i]), (int)(self.dataset.iloc[idx][1:][i + 1]))
                coordinates_int.append((x, y))

        xmin, ymin, width, height = get_bbox(landmarks=torch.tensor(coordinates_int))

        normalized_points = []
        for (x, y) in coordinates_int:
            normalized_points.append(((x - xmin.double()) / width.double(), (y - ymin.double()) / height.double()))

        return image, [(torch.tensor([xmin, ymin, width, height]), torch.tensor(normalized_points))]

