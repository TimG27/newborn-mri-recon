"""
This file contains the building blocks for the cascade-model and a few utility functions.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 128)
        self.encoder2 = DoubleConv(128, 256)
        self.encoder3 = DoubleConv(256, 512)
        self.encoder4 = DoubleConv(512, 1024)
        self.center = DoubleConv(1024, 2048)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(2048, 1024)
        self.decoder3 = DoubleConv(1024, 512)
        self.decoder2 = DoubleConv(512, 256)
        self.decoder1 = DoubleConv(256, 128)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, kernel_size=2, stride=2))
        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))
        dec4 = self.upconv4(center)
        dec4 = self.decoder4(torch.cat([self.pad(dec4, enc4), enc4], dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([self.pad(dec3, enc3), enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([self.pad(dec2, enc2), enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([self.pad(dec1, enc1), enc1], dim=1))
        output = self.final_conv(dec1)
        output2 = output + x
        return output2

    def pad(self, tensor, target):
        target_size = target.size()[2:]
        tensor_size = tensor.size()[2:]
        diff_h = target_size[0] - tensor_size[0]
        diff_w = target_size[1] - tensor_size[1]
        pad_h = (diff_h // 2, diff_h - diff_h // 2)
        pad_w = (diff_w // 2, diff_w - diff_w // 2)
        return nn.functional.pad(tensor, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
