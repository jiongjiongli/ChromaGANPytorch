from pathlib import Path


# DIRECTORY INFORMATION
DATASET = 'imagenet'
TEST_NAME ='FirstTest'
ROOT_DIR = Path('../').expanduser().resolve().absolute()
DATA_DIR = ROOT_DIR / 'DATASET'/ DATASET
OUT_DIR = ROOT_DIR / 'RESULT'/ DATASET
MODEL_DIR = ROOT_DIR / 'MODEL' / DATASET
LOG_DIR = ROOT_DIR / 'LOGS' / DATASET

TRAIN_BLACK_DIR='train_black'
TRAIN_COLOR_DIR='train_color'
TEST_BLACK_DIR='test_black'
TEST_COLOR_DIR='test_color'
INFER_BLACK_DIR='infer_black'

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 10


# TRAINING INFORMATION
NUM_EPOCHS = 10

GRADIENT_PENALTY_WEIGHT = 10

SAVE_WEIGHTS_EPOCHS_INTTERVAL = 1

MAX_TEST_NUM = 2

MAX_CHECKPOINT = 5

SAVE_IMAGE_DPI = 100

# GPU INFORMATION
def get_device():
    import torch

    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


DEVICE = get_device()


def create_dirs():
    for dir_path in (ROOT_DIR, DATA_DIR, OUT_DIR, MODEL_DIR, LOG_DIR,
                     DATA_DIR / TRAIN_BLACK_DIR,
                     DATA_DIR / TRAIN_COLOR_DIR,
                     DATA_DIR / TEST_BLACK_DIR,
                     DATA_DIR / TEST_COLOR_DIR,
                     DATA_DIR / INFER_BLACK_DIR):
        dir_path.mkdir(parents=True, exist_ok=True)

    print('Created directories')


def main():
    create_dirs()


if __name__ == '__main__':
    main()
