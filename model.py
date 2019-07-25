import os
import time

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3 , padding=1)
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3 , padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 512, 3, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1   = nn.Linear(512 * 1 * 1, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, images: Tensor) -> Tensor:
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4_1(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 512 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels)

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
