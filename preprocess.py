from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms

from utils import *

import os
import numpy as np
from PIL import Image
import json


class CityscapeDataset(data.Dataset):
    def __init__(self, args=None, transform=None, random_horizontal_flip=True,
                 random_gaussian_blur=True, random_enhance=True, split='train'):
        self.base_dir = 'dataset/{}'.format(args.dataset)
        self.images_dir = os.path.join(self.base_dir, 'Images')
        self.labels_dir = os.path.join(self.base_dir, 'Labels')
        self.annotations_dir = os.path.join(self.base_dir, 'annotations')

        with open(os.path.join(self.annotations_dir, '{}.json'.format(split))) as f:
            self.annotation = json.load(f)

        self.image_list = self.annotation['images']
        self.label_list = self.annotation['annotations']

        self.image_list.sort(key=lambda x: x['id'])
        self.label_list.sort(key=lambda x: x['image_id'])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        base_size = [2048, 1024]
        crop_size = [1024, 512]
        scale_range = [0.5, 2.0]
        ignore_mask = 255
        multi_scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        flip = True

        if transform is None:
            if split == 'train':
                transform_list = list()
                if random_horizontal_flip is True:
                    transform_list.append(RandomHorizontalFlip())
                if random_gaussian_blur is True:
                    transform_list.append(RandomGaussianBlur())
                if random_enhance is True:
                    transform_list.append(RandomEnhance())

                transform_list += [
                    RandomScaleRandomCrop(base_size, crop_size, scale_range, ignore_mask),
                    Normalize(mean=mean, std=std),
                    ToTensor()
                ]

            else:
                if split == 'val':
                    transform_list = [
                        FixedScaleCenterCrop(base_size),
                        Normalize(mean=mean, std=std),
                        ToTensor()
                    ]
                else:
                    transform_list = [
                        Normalize(mean=mean, std=std),
                        ToTensor()
                    ]

                if multi_scale is not None:
                    transform_list.append(MultiScale(multi_scale))

                if flip is True:
                    transform_list.append(Flip())
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __getitem__(self, idx):
        image_dict = self.image_list[idx]
        label_dict = self.label_list[idx]

        image = np.array(Image.open(os.path.join(self.images_dir, image_dict['file_name'])))
        label = np.array(Image.open(os.path.join(self.labels_dir, label_dict['file_name'])))

        sample = {'image': {'original_scale': image},
                  'label': {'semantic_logit': label},
                  'filename': self.annotation['annotations'][idx]['file_name']}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_list)


def load_data(args):
    train_dataset = CityscapeDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    val_dataset = CityscapeDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, drop_last=True)

    return train_loader, val_loader