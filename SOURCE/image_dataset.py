from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import config
import utils


class ImageDataset(Dataset):
    def __init__(self, black_dir, color_dir=None, include_color=True):
        self.black_dir_path = config.DATA_DIR / black_dir
        self.color_dir_path = config.DATA_DIR / color_dir if include_color else None
        self.include_color = include_color
        self.file_list = utils.find_files(self.black_dir_path,
                                          self.color_dir_path,
                                          self.include_color,
                                          ('.jpg', '.png'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        img_l, img_ab = self.read_img(file_info)
        img_l = self.transform(img_l)

        if self.include_color:
            img_ab = self.transform(img_ab)

        return img_l, img_ab, file_info

    def read_img(self, file_info):
        black_img = cv2.imread(str(file_info['black_file_path']), cv2.IMREAD_COLOR)
        black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2Lab)
        black_img = cv2.resize(black_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        img_l = black_img[:, :, :1]

        if self.include_color:
            color_img = cv2.imread(str(file_info['color_file_path']), cv2.IMREAD_COLOR)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2Lab)
            color_img = cv2.resize(color_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            img_ab = color_img[:, :, 1:]
        else:
            img_ab = np.array([], dtype=np.int8)

        return img_l, img_ab


    def transform(self, img):
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        return trans(img)


def check_dataloader():
    training_data = ImageDataset(config.TRAIN_BLACK_DIR, config.TRAIN_COLOR_DIR)
    train_dataloader = DataLoader(training_data, batch_size=config.BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=2)

    test_data = ImageDataset(config.TEST_BLACK_DIR, config.TEST_COLOR_DIR)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=2)

    infer_data = ImageDataset(config.INFER_BLACK_DIR, include_color=False)
    infer_dataloader = DataLoader(infer_data, batch_size=1, num_workers=2)

    dataloaders = {
        'train': train_dataloader,
        'test': test_dataloader,
        'infer': infer_dataloader
    }

    for data_type in ['train', 'test', 'infer']:
        batch_img_l, batch_img_ab, batch_file_infos = next(iter(dataloaders[data_type]))
        log_prefix = f'{data_type.capitalize()} batch'

        print(f'{log_prefix} size:', train_dataloader.batch_size)
        print(f'{log_prefix} number:', len(train_dataloader))
        print(f'{log_prefix} image L shape:', batch_img_l.shape)
        print(f'{log_prefix} image AB shape:', batch_img_ab.shape)
        print(f'{log_prefix} image infos:', batch_file_infos)
        print(f'{log_prefix} image L sample:', batch_img_l[0])
        print(f'{log_prefix} image AB sample:', batch_img_ab[0])


def main():
    check_dataloader()


if __name__ == '__main__':
    main()
