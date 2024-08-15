import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import config


class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()

        # VGG16 features extraction
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Update: -1 -> -8
        # self.vgg16_features = nn.Sequential(*list(vgg16.features.children())[:-1])
        self.vgg16_features = nn.Sequential(*list(vgg16.features.children())[:-8])

        # Update: Add ReLU
        self.relu = nn.ReLU()
        # Global features extraction
        # Update: Add bias, add BN eps and momentum
        # Update: Padding 1 -> (0, 1, 0, 1)
        self.global_conv1_padding = nn.ZeroPad2d((0, 1, 0, 1))
        # self.global_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)
        self.global_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0, bias=True)
        self.global_bn1 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.global_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.global_bn2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.global_conv3_padding = nn.ZeroPad2d((0, 1, 0, 1))
        # self.global_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.global_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0, bias=True)
        self.global_bn3 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.global_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.global_bn4 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)


        self.global_flatten = nn.Flatten()

        # Update: 8192 -> 512 * 7 * 7
        # self.global_fc1 = nn.Linear(8192, 1024)
        self.global_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.global_fc2 = nn.Linear(1024, 512)
        self.global_fc3 = nn.Linear(512, 256)
        # Update: Remove global_fc4
        # self.global_fc4 = nn.Linear(256, 14 * 14 * 256)

        # Update: Add global_class_flatten
        self.global_class_flatten = nn.Flatten()
        # Update: 8192 -> 512 * 7 * 7
        # self.global_class_fc1 = nn.Linear(8192, 4096)
        self.global_class_fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.global_class_fc2 = nn.Linear(4096, 4096)
        self.global_class_fc3 = nn.Linear(4096, 1000) # Same class numbers as VGG
        self.softmax = nn.LogSoftmax(dim=1)

        # Mid features extraction
        # Update: Add bias, add BN eps and momentum
        self.midlevel_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.midlevel_bn1 = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)
        self.midlevel_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.midlevel_bn2 = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)

        # Fuse features and upsample
        # Update: Add bias
        self.fusion_conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.fusion_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)

        # Update: Upsample mode='nearest'
        # self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.color_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.color_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.color_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.color_conv4 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)
        # Update: Add sigmoid
        self.sigmoid = nn.Sigmoid()

        # self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=2)


    def forward(self, x):
        # Update: Comment out permute
        # x = x.permute(0, 3, 1, 2)  # [batch_size, height, width, channels] -> [batch_size, channels, height, width]

        # VGG16 feature extraction
        vgg_features = self.vgg16_features(x)

        # Global features extraction
        # Update: conv + bn + act -> conv + act + bn
        # global_features = self.global_bn1(self.global_conv1(vgg_features))
        #
        # global_features = nn.ReLU()(global_features)
        global_features = self.global_bn1(self.relu(self.global_conv1(self.global_conv1_padding(vgg_features))))
        # global_features = self.global_bn2(self.global_conv2(global_features))
        # global_features = nn.ReLU()(global_features)
        global_features = self.global_bn2(self.relu(self.global_conv2(global_features)))

        # global_features = self.global_bn3(self.global_conv3(global_features))
        # global_features = nn.ReLU()(global_features)
        global_features = self.global_bn3(self.relu(self.global_conv3(self.global_conv3_padding(global_features))))

        # global_features = self.global_bn4(self.global_conv4(global_features))
        # global_features = nn.ReLU()(global_features)
        global_features = self.global_bn4(self.relu(self.global_conv4(global_features)))

        global_features_flat = self.global_flatten(global_features)

        # Update: Remove nn.ReLU
        # global_features2 = nn.ReLU()(self.global_fc1(global_features_flat))

        # global_features2 = nn.ReLU()(self.global_fc2(global_features2))
        # global_features2 = nn.ReLU()(self.global_fc3(global_features2))
        global_features2 = self.global_fc1(global_features_flat)

        global_features2 = self.global_fc2(global_features2)
        global_features2 = self.global_fc3(global_features2)

        # Update: global_fc4 -> repeat
        # global_features2 = self.global_fc4(global_features2).view(-1, 256, 14, 14)
        global_features2 = global_features2.unsqueeze(-1).repeat(1, 1, 28 * 28).view(-1, 256, 28, 28)

        # update: Add flatten Remove ReLU Add softmax
        # global_features_class = nn.ReLU()(self.global_class_fc1(global_features_flat))
        global_features_class = self.global_class_fc1(self.global_class_flatten(global_features))

        global_features_class = self.global_class_fc2(global_features_class)
        global_features_class = self.global_class_fc3(global_features_class)
        global_features_class = self.softmax(global_features_class)

        # Mid features extraction
        # Update: conv + bn + act -> conv + act + bn
        # midlevel_features = self.midlevel_bn1(self.midlevel_conv1(vgg_features))

        # midlevel_features = nn.ReLU()(midlevel_features)
        midlevel_features = self.midlevel_bn1(self.relu(self.midlevel_conv1(vgg_features)))
        # midlevel_features = self.midlevel_bn2(self.midlevel_conv2(midlevel_features))
        # midlevel_features = nn.ReLU()(midlevel_features)
        midlevel_features = self.midlevel_bn2(self.relu(self.midlevel_conv2(midlevel_features)))

        # Fuse global and mid features
        fused_features = torch.cat([midlevel_features, global_features2], dim=1)

        fused_features = self.relu(self.fusion_conv1(fused_features))
        fused_features = self.relu(self.fusion_conv2(fused_features))

        # Upsample and colorization
        colorization = self.upsample1(fused_features)
        colorization = self.relu(self.color_conv1(colorization))
        colorization = self.relu(self.color_conv2(colorization))
        colorization = self.upsample2(colorization)
        colorization = self.relu(self.color_conv3(colorization))
        colorization = self.sigmoid(self.color_conv4(colorization))
        colorization = self.upsample3(colorization)

        return colorization, global_features_class


def test_colorization_model():
    x = torch.randn((config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).to(config.DEVICE)
    generator = ColorizationModel().to(config.DEVICE)
    predAB, classVector = generator(x)
    print(predAB.shape)
    print(classVector.shape)
    print('test_colorization_model done')


def main():
    test_colorization_model()


if __name__ == '__main__':
    main()
