# CISSKA-LSB: Color Image Steganography using Stego Key-directed Adaptive LSB Substitution Method

## Overview

CISSKA-LSB is a Python implementation of a steganographic technique that embeds secret messages within color images using the Stego Key-directed Adaptive LSB Substitution Method.

Steganography is the practice of concealing information within other non-secret data to avoid detection. The CISSKA-LSB algorithm utilizes LSB (Least Significant Bit) substitution to hide information within the color channels of an image.

## Usage

### 1. Set Configuration

- Set the encryption key (`key`), input image filename (`filename`), and shuffle key (`key_shuffle`) in the code.

### 2. Run Encryption

- Execute the `alg_enc` function with the specified keys and the message you want to hide.

    ```python
    msg = generate_msg(1024 * 8)  # Generate a message or provide your own
    msg_bin = str_to_bin(msg)
    alg_enc(key, key_shuffle, msg_bin)
    ```

### 3. Run Decryption

- Execute the `alg_ext` function to extract the hidden message from the stego image.

    ```python
    msg_dec = alg_ext('stego.tiff', key, key_shuffle)
    print("Decrypted Message:", bin_to_str(msg_dec))
    ```

### 4. Performance Metrics

- Evaluate the performance of the steganographic process using metrics such as PSNR, SSIM, and RMSE.

    ```python
    print("PSNR:", psnr(im_org[:, :, from_:until], im_s[:, :, from_:until]))
    print("SSIM:", ssim(im_org[:, :, from_:until], im_s[:, :, from_:until], ...))
    print("RMSE:", rmse(im_org[:, :, from_:until], im_s[:, :, from_:until]))
    ```

## Dependencies

- [NumPy](https://numpy.org/)
- [Pillow (PIL Fork)](https://python-pillow.org/)
- [scikit-image](https://scikit-image.org/)

Install dependencies using:

```bash
pip install numpy pillow scikit-image
