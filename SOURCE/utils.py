from pathlib import Path
from datetime import datetime
import math
import cv2
import os
import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

import config


datetime_str_format = '%Y-%m-%d_%H-%M-%S'


def is_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return 'google.colab' in sys.modules


def is_notebook():
    try:
        from IPython import get_ipython
        # Check if IPython has 'kernel' attribute which is available in Jupyter notebooks
        return 'IPKernelApp' in get_ipython().config
    except Exception:
        return False


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        # If it's already a NumPy array, return it as is
        return x
    else:
        # Convert primitive values (e.g., int, float) to NumPy array
        return np.array(x)


def find_files(root_black_dir_path,
               root_color_dir_path,
               include_color,
               file_suffixes):
    '''Support finding files under symbolic links.
    '''
    file_list = []

    queue = collections.deque()

    if root_black_dir_path.is_dir():
        queue.append(root_black_dir_path)

    while queue:
        dir_path = queue.popleft()

        for child in dir_path.iterdir():
            if child.is_dir():
                queue.append(child)
            elif child.is_file() and child.suffix.lower() in file_suffixes:
                black_file_path = child
                file_info = {'black_file_path': str(black_file_path)}

                if include_color:
                    color_file_path = root_color_dir_path / black_file_path.relative_to(root_black_dir_path)

                    if not color_file_path.exists():
                        print(f'Ignore black-and-white file {black_file_path} because color file {color_file_path} does not exist!')
                        continue

                    file_info['color_file_path'] = str(color_file_path)

                file_list.append(file_info)

    return file_list


def save_checkpoint(epoch, generator, discriminator, G_optimizer, D_optimizer):
    print('Saving Model and Optimizer weights...')
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict' :generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer': G_optimizer.state_dict(),
        'discriminator_optimizer': D_optimizer.state_dict()
    }

    timestamp = datetime.now().strftime(datetime_str_format)
    file_name = f'epoch_{epoch:05d}_{timestamp}.pt'
    file_path = config.MODEL_DIR / file_name
    torch.save(checkpoint, str(file_path))
    print(f'Weights saved to: {file_path}')
    del checkpoint

    file_paths = list(config.MODEL_DIR.glob('epoch_*.pt'))
    file_paths.sort()

    if len(file_paths) > config.MAX_CHECKPOINT:
        file_paths[0].unlink()


def load_checkpoint(generator, discriminator=None, G_optimizer=None, D_optimizer=None):
    file_paths = list(config.MODEL_DIR.glob('epoch_*.pt'))
    file_paths.sort()

    if not file_paths:
        print('No checkpoints, the model will train from scratch.')
        return 1

    file_path = file_paths[-1]
    checkpoint = torch.load(str(file_path), map_location=config.DEVICE)
    checkpoint_epoch = checkpoint['epoch']
    generator.load_state_dict(checkpoint['generator_state_dict'])

    if discriminator is not None:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    if G_optimizer is not None:
        G_optimizer.load_state_dict(checkpoint['generator_optimizer'])

    if D_optimizer is not None:
        D_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

    print('Loaded States !!!')
    print(f'This states belong to epoch {checkpoint_epoch}.')
    print(f'so the model will train for {config.NUM_EPOCHS - checkpoint_epoch} more epochs.')
    print(f'If you want to train for more epochs, change the "NUM_EPOCHS" in config.py !!')
    return checkpoint_epoch + 1

def save_pred_output_to_file(epoch, image_index, input_img, pred_img, color_img=None, print_log=False):
    images = [input_img, pred_img]

    if color_img is not None:
        images.append(color_img)

    titles = ['Input', 'Pred', 'Real Color']

    rows = 1
    columns = len(images)

    fig , axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    fig.suptitle(f'Epoch: {epoch} Image index: {image_index}', fontsize=24)

    for image, title, ax in zip(images, titles, axes.ravel()):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

    timestamp = datetime.now().strftime(datetime_str_format)
    file_name = f'sample_preds_{epoch:05d}_{image_index:05d}_{timestamp}.png'
    file_path = config.OUT_DIR / file_name

    plt.savefig(str(file_path), dpi=config.SAVE_IMAGE_DPI)
    plt.close()

    if print_log:
        print(f'Saved predict output to {file_path}')


def merge_pred_images():
    file_paths = list(config.OUT_DIR.glob('sample_preds_*.png'))
    file_paths.sort()

    if not file_paths:
        return

    width_inchs = []
    height_inchs = []

    for file_path in file_paths:
        # Load the image using PIL
        with Image.open(str(file_path)) as image:
            # Get the size in pixels
            width_px, height_px = image.size

            # Calculate size in inches
            width_inch = int(math.ceil(width_px / config.SAVE_IMAGE_DPI))
            width_inchs.append(width_inch)
            height_inch = int(math.ceil(height_px / config.SAVE_IMAGE_DPI))
            height_inchs.append(height_inch)

    rows = len(file_paths)
    columns = 1
    max_width_inch = max(width_inchs)
    total_height_inch = sum(height_inchs)

    fig, axes = plt.subplots(rows, columns, figsize=(max_width_inch, total_height_inch))

    for file_path, ax in zip(file_paths, axes.ravel()):
        # Load the image using PIL
        with Image.open(str(file_path)) as image:
            ax.imshow(image)
            ax.axis('off')

    file_name = f'merged.png'
    file_path = config.OUT_DIR / file_name

    plt.savefig(str(file_path), dpi=config.SAVE_IMAGE_DPI)
    plt.close()
    print(f'Merged prediction images to {file_path}')


def save_loss_data_to_file(loss_data):
    timestamp = datetime.now().strftime(datetime_str_format)
    file_name = f'loss_data_{timestamp}.csv'
    file_path = config.LOG_DIR / file_name

    df = pd.DataFrame(data=loss_data)
    df.to_csv(file_path)
    print(f'Saved loss data to {file_path}')


def main():
    merge_pred_images()


if __name__ == '__main__':
    main()
