from torch.utils.data import DataLoader

import config
import utils
from predict import run_predict
from image_dataset import ImageDataset
from generator_model import ColorizationModel
from discriminator_model import Discriminator


def run_test():
    # Create directories
    config.create_dirs()

    # DataLoader
    test_data = ImageDataset(config.TEST_BLACK_DIR, config.TEST_COLOR_DIR)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=2)

    # Load discriminator and generator model
    generator = ColorizationModel().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)

    next_epoch = utils.load_checkpoint(generator)
    epoch = next_epoch - 1

    if epoch == 0:
        print('Error! No checkpoint exists')
        return

    run_predict(epoch, test_dataloader, generator)
    print('Completed test!')


def main():
    run_test()


if __name__ == '__main__':
    main()
