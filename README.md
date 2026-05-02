# SPI-PGAN

Physics-driven GAN for single-pixel imaging reconstruction.

## What this does

This code reconstructs 32×32 images from single-pixel detector measurements. It takes a set of modulation patterns and corresponding bucket detector signals, then uses a physics-driven GAN to produce high-quality images.

Compared to traditional methods (DGI, TVAL3), it gives sharper edges and less noise.

## Requirements

- Python 3.8+
- PyTorch 1.10+

```bash
pip install -r requirements.txt
# SPI-PGAN
