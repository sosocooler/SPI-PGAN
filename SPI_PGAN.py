"""
SPI-PGAN: Training Script
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

from SPI_PGAN_model import PGANModel


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    # Data paths
    PATTERN_FILE = './Patterns.txt'
    DETECTION_FILE = './Diff_Signals.txt'
    RESULTS_DIR = './results/'

    # Image parameters
    IMG_W = 32
    IMG_H = 32
    SAMPLING_RATE = 0.5

    # Training parameters
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
    h_kernel = torch.tensor([[-1, 2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 1, 3)
    v_kernel = torch.tensor([[-1], [2], [-1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 1)
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
# Training Function
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

    if y_total.ndim == 2:
        print(f"y_total is 2D with shape {y_total.shape}, taking first column")
        y_total = y_total[:, 0]

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

    model = PGANModel(cfg.IMG_W, cfg.IMG_H, num_patterns).to(device)
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