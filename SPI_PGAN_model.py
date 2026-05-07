"""
SPI-PGAN: Physics-driven GAN for Robust High-speed Single-pixel Imaging
Model Architecture
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
               norm=True, activation=True, transpose=False):
    """Convolution or transposed convolution block"""
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride, padding, bias=not norm))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, bias=not norm))
    if norm:
        layers.append(nn.BatchNorm2d(out_channels, affine=True))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def res_block(channels):
    """Residual block"""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels, affine=True)
    )


class AntiAliasBlock(nn.Module):
    """Anti-aliasing block for downsampling"""
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(AntiAliasBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.blur_kernel = nn.Parameter(
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2),
            requires_grad=False
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels)
        self.norm = nn.BatchNorm2d(channels, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        blurred_list = []
        for c in range(ch):
            x_c = x[:, c:c + 1, :, :]
            blurred_c = F.conv2d(x_c, self.blur_kernel, stride=self.stride, padding=self.padding)
            blurred_list.append(blurred_c)
        x_blurred = torch.cat(blurred_list, dim=1)
        x_blurred = self.conv(x_blurred)
        x_blurred = self.norm(x_blurred)
        x_blurred = self.activation(x_blurred)
        return x_blurred


class Generator(nn.Module):
    """Generator network"""
    def __init__(self, img_W, img_H, num_patterns, ngf=64):
        super(Generator, self).__init__()
        self.img_W = img_W
        self.img_H = img_H
        self.num_patterns = num_patterns
        self.ngf = ngf

        self.downsample_initial = nn.Sequential(
            nn.Conv2d(1, ngf, kernel_size=7, padding=3),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )

        self.enc1 = conv_block(ngf, ngf * 2, norm=True)
        self.enc2 = conv_block(ngf * 2, ngf * 4, norm=True)
        self.enc3 = conv_block(ngf * 4, ngf * 8, norm=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 8, affine=True),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[res_block(ngf * 8) for _ in range(2)])
        self.anti_alias_bottleneck = AntiAliasBlock(ngf * 8)

        self.dec3 = conv_block(ngf * 8 + ngf * 8, ngf * 4, norm=True, transpose=True)
        self.anti_alias_dec3 = AntiAliasBlock(ngf * 4)
        self.dec2 = conv_block(ngf * 4 + ngf * 4, ngf * 2, norm=True, transpose=True)
        self.anti_alias_dec2 = AntiAliasBlock(ngf * 2)
        self.dec1 = conv_block(ngf * 2 + ngf * 2, ngf, norm=True, transpose=True)
        self.anti_alias_dec1 = AntiAliasBlock(ngf)

        self.final_layer = nn.Sequential(
            nn.Conv2d(ngf + ngf, ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, inpt, real_A):
        x0 = self.downsample_initial(inpt)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x_b = self.bottleneck(x3)
        x_b = self.res_blocks(x_b)
        x_b = self.anti_alias_bottleneck(x_b)

        x = torch.cat([x_b, x3], dim=1)
        x = self.dec3(x)
        x = self.anti_alias_dec3(x)

        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.anti_alias_dec2(x)

        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        x = self.anti_alias_dec1(x)

        x = torch.cat([x, x0], dim=1)
        out_x_raw = self.final_layer(x)

        out_x_pos = (out_x_raw + 1.0) / 2.0
        out_x_pos = torch.clamp(out_x_pos, min=0.0)
        max_val = torch.amax(out_x_pos, dim=[1, 2, 3], keepdim=True)
        max_val = torch.clamp(max_val, min=1e-8)
        out_x_normalized = out_x_pos / max_val

        out_y = torch.sum(out_x_normalized * real_A, dim=[2, 3], keepdim=True)

        mean_x = out_x_pos.mean(dim=[1, 2, 3], keepdim=True)
        var_x = out_x_pos.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        mean_y = out_y.mean(dim=[1, 2, 3], keepdim=True)
        var_y = out_y.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        out_x_norm = (out_x_pos - mean_x) / torch.sqrt(var_x + 1e-8)
        out_y_norm = (out_y - mean_y) / torch.sqrt(var_y + 1e-8)

        return out_x_norm, out_y_norm, out_x_raw


class Discriminator(nn.Module):
    """Discriminator network"""
    def __init__(self, img_W, img_H, ndf=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, img):
        return self.model(img)


class PGANModel(nn.Module):
    """Complete PGAN model wrapper"""
    def __init__(self, img_W, img_H, num_patterns, ngf=64, ndf=64):
        super(PGANModel, self).__init__()
        self.generator = Generator(img_W, img_H, num_patterns, ngf)
        self.discriminator = Discriminator(img_W, img_H, ndf)

    def forward(self, inpt, real_A):
        return self.generator(inpt, real_A)

    def discriminate(self, img):
        return self.discriminator(img)