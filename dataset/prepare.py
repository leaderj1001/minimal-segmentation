import torch

import os
from PIL import Image
import numpy as np
import shutil
import pickle
import cv2
import glob


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def encode_semantic_label(label, ignore_mask=255):
    color_map = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                  [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                  [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    encoded_label = np.ones((*label.shape[:2],)) * ignore_mask
    for i in range(len(color_map)):
        encoded_label[np.all(label == color_map[i], axis=-1)] = i

    return encoded_label.astype(np.int32)


def decode_semantic_label(label):
    color_map = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                 [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    if type(label) == np.ndarray:
        if len(label.shape) == 3:
            label = np.argmax(label, axis=0)

        decoded_label = np.zeros((*label.shape, 3))
        for i in range(len(color_map)):
            decoded_label[label == i] = color_map[i]

        return decoded_label

    elif type(label) == torch.Tensor:
        if len(label.shape) == 3:
            label = torch.argmax(label, dim=0)
        color_map = torch.tensor(color_map)

        decoded_label = torch.zeros((*label.shape, 3)).long()
        for i in range(len(color_map)):
            decoded_label[label == i] = color_map[i]

        return decoded_label


def main(args):
    base_dir = './dataset/{}'.format(args.dataset)
    image_dir = os.path.join(base_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit')
    gt_dir = os.path.join(base_dir, 'gtFine_trainvaltest', 'gtFine')
    split = ['train', 'val', 'test']

    prepared_image = '{}/Image'.format(base_dir)
    prepared_label = '{}/Label'.format(base_dir)
    prepared_annotation = '{}/Annotation'.format(base_dir)

    if not os.path.isdir(prepared_image):
        os.mkdir(prepared_image)
    if not os.path.isdir(prepared_label):
        os.mkdir(prepared_label)
    if not os.path.isdir(prepared_annotation):
        os.mkdir(prepared_annotation)

    for _ in split:
        sample = {}
        count = 0
        base_image = os.path.join(image_dir, _)
        for (root, dirs, files) in os.walk(base_image):
            for file in files:
                id = int(file.split('_')[1])
                sample[count] = {
                    'filename': file,
                    'id': id,
                    'width': 2048,
                    'height': 1024,
                }
                b_image = os.path.join(root, file)
                s_image = os.path.join(prepared_image, file)
                shutil.copyfile(b_image, s_image)
                count += 1

        base_label = os.path.join(gt_dir, _)
        label_list = []
        dirs = os.listdir(base_label)
        count = 0
        for dir in dirs:
            _d = os.path.join(base_label, dir)
            label_list.extend(glob.glob(os.path.join(_d, '*color.png')))
        for l in label_list:
            file_name = os.path.basename(l).replace('color', 'label')
            img = np.array(Image.open(l).convert('RGB'))
            label = encode_semantic_label(img)
            s_image = os.path.join(prepared_label, file_name)
            sample[count]['label_name'] = file_name
            cv2.imwrite(s_image, label)
            count += 1
        save_pickle(sample, os.path.join(prepared_annotation, '{}.pkl'.format(_)))


if __name__ == '__main__':
    from config import load_config
    args = load_config()
    main(args)
