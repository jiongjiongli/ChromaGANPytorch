from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

import config
import utils
from image_dataset import ImageDataset
from generator_model import ColorizationModel
from discriminator_model import Discriminator
from predict import run_predict


def wasserstein_loss(inputs, real_or_fake):
    """
    Wasserstein loss: https://arxiv.org/abs/1701.07875
    :param inputs: input to the loss function
    :return Wasserstein loss
    """
    if real_or_fake:
        return -torch.mean(inputs)
    else:
        return torch.mean(inputs)


def partial_gradient_penalty_loss(y_pred, averaged_samples, gradient_penalty_weight):
    """
    Computes partial gradient penalty loss.
    :param target: target image
    :param averaged_samples: weighted real / fake samples
    :param gradient_penalty_weight: weight of the gradient penalty
    :return: partial gradient penalty
    """
    gradients = torch.autograd.grad(
        y_pred, averaged_samples,
        grad_outputs=torch.ones(y_pred.size(), device=config.DEVICE),
        create_graph=True, retain_graph=True
        )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_l2_norm = gradients.norm(2, dim=1)
    gradient_penalty = gradient_penalty_weight * torch.square(1 - gradient_l2_norm)
    gradient_penalty = gradient_penalty.mean()
    return gradient_penalty


def train():
    # Create directories
    config.create_dirs()

    # Setup TensorBoard
    tensor_board_writer = SummaryWriter(str(config.LOG_DIR))

    # DataLoader
    training_data = ImageDataset(config.TRAIN_BLACK_DIR, config.TRAIN_COLOR_DIR)
    train_dataloader = DataLoader(training_data, batch_size=config.BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=2)

    test_data = ImageDataset(config.TEST_BLACK_DIR, config.TEST_COLOR_DIR)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=2)

    # Load discriminator and generator model
    generator = ColorizationModel().to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)

    G_optimizer = optim.Adam(generator.parameters(), lr=0.00002,
                             betas=(0.5, 0.999), eps=1e-07)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002,
                             betas=(0.5, 0.999), eps=1e-07)

    epoch = utils.load_checkpoint(generator, discriminator, G_optimizer, D_optimizer)

    vgg_model_f = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(config.DEVICE)
    vgg_model_f.eval()

    # init loss function
    KLDivergence = nn.KLDivLoss(reduction='batchmean')
    MSE = nn.MSELoss()

    # loss log
    loss_items = ('Epoch',         'Batch',
                  'G_loss',        'D_loss',
                  'G_loss_MSE',    'G_loss_KLD',    'G_loss_W',
                  'D_loss_w_real', 'D_loss_w_fake', 'D_loss_gp'
                  )

    loss_data = {loss_item: [] for loss_item in loss_items}

    if utils.is_colab() or utils.is_kaggle() or utils.is_notebook():
        progress_bar_ncols = 250
    else:
        progress_bar_ncols = None

    global_step = 0

    while epoch <= config.NUM_EPOCHS:
        description = f'Epoch {epoch}/{config.NUM_EPOCHS}'
        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc=description,
                            ncols=progress_bar_ncols)

        for idx, (input_L, real_AB, _) in progress_bar:
            global_step += 1

            input_L = input_L.to(config.DEVICE)
            real_AB = real_AB.to(config.DEVICE)
            real_LAB = torch.cat([input_L, real_AB], dim=1)

            input_L3 = input_L.repeat(1, 3, 1, 1)

            with torch.no_grad():
                real_class = F.softmax(vgg_model_f(input_L3), dim=1)

            # ----------------- Train the discriminator -----------------
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False  # to avoid computation

            D_optimizer.zero_grad()

            real_disc_output = discriminator(real_LAB)

            fake_AB, fake_class = generator(input_L3)
            fake_LAB = torch.cat([input_L, fake_AB], dim=1)
            fake_disc_output = discriminator(fake_LAB)

            weights = torch.randn((config.BATCH_SIZE, 1, 1, 1), device=config.DEVICE)
            averaged_samples = (weights * real_AB) + ((1 - weights) * fake_AB)
            averaged_samples = averaged_samples.requires_grad_(True)
            avg_img = torch.cat([input_L, averaged_samples], dim=1)
            avg_dist_output = discriminator(avg_img)

            D_loss_w_real = wasserstein_loss(real_disc_output, True)
            D_loss_w_fake = wasserstein_loss(fake_disc_output, False)
            D_loss_gp = partial_gradient_penalty_loss(avg_dist_output, averaged_samples, config.GRADIENT_PENALTY_WEIGHT)
            D_loss = D_loss_w_real + D_loss_w_fake + D_loss_gp
            D_loss.backward()
            D_optimizer.step()

            # ----------------- Train the generator -----------------
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            G_optimizer.zero_grad()
            fake_AB, fake_class = generator(input_L3)
            fake_LAB = torch.cat([input_L, fake_AB], dim=1)

            fake_disc_output = discriminator(fake_LAB)

            G_loss_MSE = MSE(fake_AB, real_AB)
            G_loss_KLD = KLDivergence(
                fake_class,
                real_class
                ) * 0.003
            G_loss_W = wasserstein_loss(fake_disc_output, True) * 0.1
            G_loss = G_loss_MSE + G_loss_KLD + G_loss_W
            G_loss.backward()
            G_optimizer.step()

            # ----------------- Log the trainning process  -----------------
            loss_values = (epoch,         idx,
                           G_loss,        D_loss,
                           G_loss_MSE,    G_loss_KLD,    G_loss_W,
                           D_loss_w_real, D_loss_w_fake, D_loss_gp
                           )

            loss_str_dict = {}

            for loss_item, loss_value in zip(loss_items, loss_values):
                loss_data[loss_item].append(utils.to_numpy(loss_value))

                if loss_item in ['Epoch', 'Batch']:
                    continue

                loss_str = f'{loss_value:.4f}'
                loss_str_dict[loss_item] = loss_str

                tensor_board_writer.add_scalar(f'Loss/{loss_item}', loss_value, global_step)

            progress_bar.set_postfix(ordered_dict=loss_str_dict)

            if idx + 1 == len(train_dataloader):
                if (epoch % config.SAVE_WEIGHTS_EPOCHS_INTTERVAL == 0) or (epoch + 1 == config.NUM_EPOCHS):
                    run_predict(epoch, test_dataloader, generator, config.MAX_TEST_NUM)
                    utils.save_checkpoint(epoch, generator, discriminator, G_optimizer, D_optimizer)

        tensor_board_writer.flush()
        epoch += 1

    utils.save_loss_data_to_file(loss_data)
    tensor_board_writer.close()
    print('Completed train!')

def main():
    train()


if __name__ == '__main__':
    main()
