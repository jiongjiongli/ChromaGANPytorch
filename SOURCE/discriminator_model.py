import torch
import torch.nn as nn

import config


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Update: Add bias
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)

        # Update: Padding 1 -> (1, 2, 1, 2)
        self.conv4_padding = nn.ZeroPad2d((1, 2, 1, 2))
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0, bias=True)

        # Update: Padding 1 -> (1, 2, 1, 2)
        self.conv5_padding = nn.ZeroPad2d((1, 2, 1, 2))
        # self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True)

        # Update: 02. -> 0.3, inplace=False
        # self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.3)

    # Update: Update args
    # def forward(self, input_ab, input_l):
    def forward(self, x):
        # Update: Comment out cat
        # x = torch.cat([input_l, input_ab], dim=1)

        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.conv4_padding(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.conv5_padding(x)
        x = self.conv5(x)

        return x


def test_discriminator():
    x = torch.randn((config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).to(config.DEVICE)
    discriminator = Discriminator().to(config.DEVICE)
    pred = discriminator(x)
    print(pred.shape)
    print('test_discriminator done')


def main():
    test_discriminator()


if __name__ == '__main__':
    main()
