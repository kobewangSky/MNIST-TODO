from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN
        self.mode = mode
        self._mnist = datasets.MNIST(path_to_data_dir, train=is_train, download=True)
        print("1")

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.mode == Dataset.Mode.TRAIN:
            return self._mnist.train_data[index], self._mnist.train_labels[index]
        else:
            return self._mnist.test_data[index], self._mnist.test_labels[index]

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
        return image
