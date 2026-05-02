"""
SPI-PGAN: Physics-driven GAN for Robust High-speed Single-pixel Imaging
========================================================================
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import gc


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data
    PATTERN_FILE = './PP.txt'
    DETECTION_FILE = './demo.txt'
    RESULTS_DIR = './results/'

    # Image
    IMG_W = 32
    IMG_H = 32
    SAMPLING_RATE = 0.5

    # Training
    STEPS = 101
    BATCH_SIZE = 1
    LR_G = 0.002
    LR_D = 0.001
    BETA1 = 0.5
    BETA2 = 0.999

    # Loss weights
    LAMBDA_L1 = 100.0
    LAMBDA_TV = 6e-2
    LAMBDA_H_NOISE = 4e-2
    LAMBDA_V_NOISE = 4e-2


# ============================================================================
# Network Components
# ============================================================================

def init_weights(m):
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
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels, affine=True)
    )


class AntiAliasBlock(nn.Module):
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
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0,
                              groups=channels)
        self.norm = nn.BatchNorm2d(channels, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        ch = x.size(1)
        blurred_list = []
        for c in range(ch):
            x_c = x[:, c:c + 1, :, :]
            blurred_c = F.conv2d(x_c, self.blur_kernel, stride=self.stride,
                                 padding=self.padding)
            blurred_list.append(blurred_c)
        x_blurred = torch.cat(blurred_list, dim=1)
        x_blurred = self.conv(x_blurred)
        x_blurred = self.norm(x_blurred)
        x_blurred = self.activation(x_blurred)
        return x_blurred


class Generator(nn.Module):
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


class PGAN(nn.Module):
    def __init__(self, img_W, img_H, num_patterns, ngf=64, ndf=64):
        super(PGAN, self).__init__()
        self.generator = Generator(img_W, img_H, num_patterns, ngf)
        self.discriminator = Discriminator(img_W, img_H, ndf)

    def forward(self, inpt, real_A):
        return self.generator(inpt, real_A)

    def discriminate(self, img):
        return self.discriminator(img)


# ============================================================================
# Loss Functions
# ============================================================================

def total_variation_loss(x):
    x_pos = (x + 1.0) / 2.0
    x_pos = torch.clamp(x_pos, min=0.0)
    tv_h = torch.sum(torch.abs(x_pos[:, :, 1:, :] - x_pos[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(x_pos[:, :, :, 1:] - x_pos[:, :, :, :-1]))
    return tv_h + tv_w


def noise_suppression_loss(x, h_weight, v_weight):
    x_flat = x.view(-1, 1, x.size(2), x.size(3))
    h_kernel = torch.tensor([[-1, 2, -1]], dtype=torch.float32,
                            device=x.device).view(1, 1, 1, 3)
    v_kernel = torch.tensor([[-1], [2], [-1]], dtype=torch.float32,
                            device=x.device).view(1, 1, 3, 1)
    h_diff = F.conv2d(x_flat, h_kernel, padding=(0, 1))
    v_diff = F.conv2d(x_flat, v_kernel, padding=(1, 0))
    return h_weight * torch.mean(h_diff ** 2) + v_weight * torch.mean(v_diff ** 2)


# ============================================================================
# DGI Reconstruction
# ============================================================================

def dgi_reconstruct(patterns, detections, img_W, img_H, num_patterns):
    B_aver, SI_aver, R_aver, RI_aver = 0, 0, 0, 0
    count = 0
    for i in range(num_patterns):
        pattern = patterns[:, :, i]
        count += 1
        B_r = detections[i]
        SI_aver = (SI_aver * (count - 1) + pattern * B_r) / count
        B_aver = (B_aver * (count - 1) + B_r) / count
        R_aver = (R_aver * (count - 1) + np.sum(pattern)) / count
        RI_aver = (RI_aver * (count - 1) + np.sum(pattern) * pattern) / count
        DGI = SI_aver - B_aver / R_aver * RI_aver
    DGI = DGI * 255 / np.max(DGI)
    return DGI


# ============================================================================
# Training
# ============================================================================

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_basename = os.path.splitext(os.path.basename(cfg.DETECTION_FILE))[0]
    result_save_path = os.path.join(cfg.RESULTS_DIR, input_basename)
    os.makedirs(result_save_path, exist_ok=True)
    print(f"Results will be saved in: {result_save_path}")

    num_patterns = int(np.round(cfg.IMG_W * cfg.IMG_H * cfg.SAMPLING_RATE))

    A_total = np.loadtxt(cfg.PATTERN_FILE)
    y_total = np.loadtxt(cfg.DETECTION_FILE)

    A_real = np.zeros([cfg.IMG_W, cfg.IMG_H, num_patterns])
    y_real = np.zeros(num_patterns)
    for i in range(num_patterns):
        temp = A_total[i, :]
        A_real[:, :, i] = temp.reshape(cfg.IMG_W, cfg.IMG_H)
        y_real[i] = y_total[i]

    print(f"num_patterns = {num_patterns}")
    print('Data Loading Finished')

    print('DGI reconstruction...')
    DGI = dgi_reconstruct(A_real, y_real, cfg.IMG_W, cfg.IMG_H, num_patterns)
    print('DGI Reconstruction Finished')

    plt.imshow(DGI, cmap='gray')
    plt.title('DGI Reconstruction')
    plt.show()

    dgi_to_save = DGI - np.min(DGI)
    dgi_to_save = dgi_to_save * 255 / np.max(dgi_to_save + 1e-8)
    dgi_img = Image.fromarray(dgi_to_save.astype('uint8')).convert('L')
    dgi_img.save(os.path.join(result_save_path, f'{input_basename}_DGI.bmp'))

    y_real_tensor = torch.tensor(y_real, dtype=torch.float32).view(1, num_patterns, 1, 1).to(device)
    A_real_tensor = torch.tensor(A_real, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    DGI_tensor = torch.tensor(DGI, dtype=torch.float32).view(1, 1, cfg.IMG_W, cfg.IMG_H).to(device)

    DGI_tensor = (DGI_tensor - DGI_tensor.mean()) / DGI_tensor.std()
    y_real_tensor = (y_real_tensor - y_real_tensor.mean()) / y_real_tensor.std()
    A_real_tensor = (A_real_tensor - A_real_tensor.mean()) / A_real_tensor.std()

    model = PGAN(cfg.IMG_W, cfg.IMG_H, num_patterns).to(device)
    gen = model.generator
    disc = model.discriminator

    optimizer_G = torch.optim.Adam(gen.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    DGI_temp0 = DGI_tensor.squeeze().cpu().numpy().T

    print('Training PGAN...')
    collage_images, collage_titles = [], []

    for step in range(cfg.STEPS):
        disc.train()
        optimizer_D.zero_grad()

        gen_x_norm, gen_y_norm, gen_x_raw = gen(DGI_tensor, A_real_tensor)
        fake_output = disc(gen_x_raw.detach())

        real_labels = torch.ones_like(fake_output).to(device)
        fake_labels = torch.zeros_like(fake_output).to(device)

        errD_real = criterion_GAN(disc(DGI_tensor), real_labels)
        errD_fake = criterion_GAN(fake_output, fake_labels)
        errD = (errD_real + errD_fake) * 0.5
        errD.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        optimizer_D.step()

        gen.train()
        optimizer_G.zero_grad()

        gen_x_norm, gen_y_norm, gen_x_raw = gen(DGI_tensor, A_real_tensor)
        fake_output_for_G = disc(gen_x_raw)
        real_labels_for_G = torch.ones_like(fake_output_for_G).to(device)

        errG_adv = criterion_GAN(fake_output_for_G, real_labels_for_G)
        errG_l1_y = criterion_L1(gen_y_norm, y_real_tensor)
        errG_TV = cfg.LAMBDA_TV * total_variation_loss(gen_x_raw)
        errG_noise = noise_suppression_loss(gen_x_raw, cfg.LAMBDA_H_NOISE, cfg.LAMBDA_V_NOISE)

        errG = errG_adv + cfg.LAMBDA_L1 * errG_l1_y + errG_TV + errG_noise
        errG.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        optimizer_G.step()

        if step % 5 == 0:
            print(f'Step: {step}, D_loss: {errD.item():.4f}, G_loss: {errG.item():.4f}, '
                  f'Adv_loss: {errG_adv.item():.4f}, L1_y_loss: {errG_l1_y.item():.4f}, '
                  f'TV_loss: {errG_TV.item():.6f}, Noise_loss: {errG_noise.item():.6f}')

            gen_x_display = ((gen_x_raw.squeeze().detach().cpu().numpy() + 1.0) / 2.0).T
            x_out_to_save = gen_x_display.T
            x_out_to_save = x_out_to_save - np.min(x_out_to_save)
            x_out_to_save = x_out_to_save * 255 / np.max(x_out_to_save + 1e-8)
            x_out_img = Image.fromarray(x_out_to_save.astype('uint8')).convert('L')
            x_out_img.save(os.path.join(result_save_path, f'{input_basename}_{step}.bmp'))

            collage_images.append(DGI_temp0)
            collage_titles.append(f'DGI (Step {step})')
            collage_images.append(gen_x_display)
            collage_titles.append(f'PGAN (Step {step})')

        if step % 20 == 0 and step > 0:
            num_cols = 2
            num_rows = len(collage_images) // num_cols
            if num_rows > 0:
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 3 * num_rows))
                if num_rows == 1:
                    axes = axes.reshape(1, -1)
                for i in range(len(collage_images)):
                    row, col = i // num_cols, i % num_cols
                    axes[row, col].imshow(collage_images[i], cmap='gray')
                    axes[row, col].set_title(collage_titles[i])
                    axes[row, col].axis('off')
                plt.tight_layout()
                plt.show()

    print('Training Finished!')

    del model, gen, disc, optimizer_G, optimizer_D
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleaned.")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    cfg = Config()
    train(cfg)