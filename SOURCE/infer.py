from torch.utils.data import DataLoader

import config
import utils
from predict import run_predict
from image_dataset import ImageDataset
from generator_model import ColorizationModel
from discriminator_model import Discriminator


def run_infer():
    # Create directories
    config.create_dirs()

    # DataLoader
    infer_data = ImageDataset(config.INFER_BLACK_DIR, include_color=False)
    infer_dataloader = DataLoader(infer_data, batch_size=1, num_workers=2)

    # Load discriminator and generator model
    generator = ColorizationModel().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)

    next_epoch = utils.load_checkpoint(generator)
    epoch = next_epoch - 1

    if epoch == 0:
        print('Error! No checkpoint exists')
        return

    run_predict(epoch, infer_dataloader, generator)
    print('Completed inference!')


def main():
    run_infer()


if __name__ == '__main__':
    main()
