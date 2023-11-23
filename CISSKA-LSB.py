import random
import textwrap
from math import sqrt

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Set the encryption key, input filename, and shuffle key
key = '32741586'
filename = 'building.tiff'
key_shuffle = [(2, 5), (1, 7), (6, 4), (3, 0)]


def int_to_bin(k):
    kk = ''
    for i in range(len(k)):
        bin_k = bin(int(k[i]))[2:].zfill(8)
        kk = kk + bin_k
    return kk


def str_to_bin(s):
    kk = ''
    for i in range(len(s)):
        bin_k = bin(ord(s[i]))[2:].zfill(8)
        kk = kk + bin_k
    return kk


def bin_to_str(b):
    res = ''
    txt = textwrap.wrap(b, 8)
    for b in txt:
        res += chr(int(b, 2))
    return res


def x0r(a, b):
    ka = ''
    for i in range(len(a)):
        if a[i] == b[i % len(b)]:
            ka = ka + '0'
        elif a[i] != b[i % len(b)]:
            ka = ka + '1'
    return ka


def shuffleTLEA(lst_key, ka):
    txt = textwrap.wrap(ka, 8)
    lst = []
    for i in txt:
        s = i
        for k in lst_key:
            s = swap(s, k[0], k[1])
        lst.append(s)
    return ''.join(lst)


def swap(c, i, j):
    c = list(c)
    c[i], c[j] = c[j], c[i]
    return ''.join(c)


def build_1(n):
    return ''.join(['1' for i in range(n)])


def TLEA(k, shuffle_key):
    bin_key = int_to_bin(k)
    xor_res = x0r(bin_key, build_1(len(bin_key)))
    return shuffleTLEA(shuffle_key, xor_res)


def block_div(msg):
    str_lst = textwrap.wrap(msg, 8)
    arr_msg = ['', '', '', '']
    for s in str_lst:
        for i in range(len(s) // 2):
            arr_msg[i] += s[i] + s[-1 - i]
    return ''.join(arr_msg)


def inverse_block_div(msg):
    str_lst = textwrap.wrap(msg, len(msg) // 4)
    arr_byte = ['' for i in range(len(msg) // 8)]
    for b in range(len(arr_byte)):
        for s in str_lst:
            arr_byte[b] += s[b * 2]
        for s in str_lst[::-1]:
            arr_byte[b] += s[b * 2 + 1]
    return ''.join(arr_byte)


def shuffleMLEA(key, MM):
    MMM = ''
    MM_lst = textwrap.wrap(MM, 8)
    for k in MM_lst:
        for i in range(len(key)):
            MMM += k[int(key[i]) - 1]
    return MMM


def inverseShuffleMLEA(key, MM):
    MMM = ''
    MM_lst = textwrap.wrap(MM, 8)
    for k in MM_lst:
        tmp = [0 for j in range(8)]
        for i in range(len(key)):
            tmp[int(key[i]) - 1] = k[i]
        MMM += ''.join(tmp)
    return MMM


def MLEA(message, keyTLEA, key):
    xor1_res = x0r(message, build_1(len(message)))
    div_bl = block_div(xor1_res)
    shaff_mlea = shuffleMLEA(key, div_bl)
    xorWithkey = x0r(shaff_mlea, keyTLEA)
    return xorWithkey


def MLEA_dec(message, keyTLEA, key):
    xorWithkey = x0r(message, keyTLEA)
    shaff_mlea = inverseShuffleMLEA(key, xorWithkey)
    div_bl = inverse_block_div(shaff_mlea)
    xor1_res = x0r(div_bl, build_1(len(message)))
    return xor1_res


def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def permute(arr, seed):
    index = np.arange(arr.shape[0])
    new_index = np.random.RandomState(seed=seed).permutation(index)
    lst = []
    for i in new_index:
        lst.append(arr[i])
    return np.array(lst)


def repermute(arr, seed):
    index = np.arange(arr.shape[0])
    new_index = np.random.RandomState(seed=seed).permutation(index)
    lst = [0 for i in range(len(arr))]
    for i in range(len(new_index)):
        lst[new_index[i]] = (arr[i])
    return np.array(lst)


def alg_enc(key_skTLEA, key_shuffle, msg):
    global filename
    im = Image.open(filename)
    imarray = np.array(im)
    imarray = permute(imarray, 12930)
    img = Image.fromarray(imarray.astype('uint8'), 'RGB')
    shape = img.size
    r = imarray[:, :, 0].ravel()
    g = imarray[:, :, 1].ravel()
    b = imarray[:, :, 2].ravel()

    c = str_to_bin(str(len(msg) // 8))
    for i in range(len(c)):
        if c[i] == '0':
            r[-i - 1] = clear_bit(r[-i - 1], 0)
        else:
            r[-i - 1] = set_bit(r[-i - 1], 0)

    key_ESK = TLEA(key_skTLEA, key_shuffle)
    M_esi = MLEA(msg, key_ESK, key_skTLEA)

    count = 0
    i = 0
    while count <= (len(M_esi) - 1):
        RLSB = r[count] & 1
        ENC_SEC_KEYBIT = key_ESK[i]
        if str(RLSB) != ENC_SEC_KEYBIT:
            if M_esi[count] == '0':
                g[count] = clear_bit(g[count], 0)
            elif M_esi[count] == '1':
                g[count] = set_bit(g[count], 0)
        else:
            if M_esi[count] == '0':
                b[count] = clear_bit(b[count], 0)
            elif M_esi[count] == '1':
                b[count] = set_bit(b[count], 0)

        i += 1
        count += 1
        if i >= len(key_ESK):
            i = 0
    new_img = np.dstack((r, g, b)).reshape((shape[0], shape[1], 3))
    new_img = repermute(new_img, 12930)
    new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    new_img.save('stego.tiff')


def alg_ext(I_s, key_stkSTEGANO, key_shuffle):
    im_ex = Image.open(I_s)
    im_ex_array = np.array(im_ex)
    im_ex_array = permute(im_ex_array, 12930)
    r = im_ex_array[:, :, 0].ravel()
    g = im_ex_array[:, :, 1].ravel()
    b = im_ex_array[:, :, 2].ravel()
    msg_size = ''
    for i in range(32):
        msg_size += str(r[-1 - i] & 1)
    msg_size = int(bin_to_str(msg_size)) * 8
    key_ESK = TLEA(key_stkSTEGANO, key_shuffle)
    i = 0
    count = 0
    msg = ''
    while msg_size > count:
        RLSB = r[count] & 1
        ENC_SEC_KEYBIT = key_ESK[i]
        if str(RLSB) != ENC_SEC_KEYBIT:
            msg += str(g[count] % 2)
        else:
            msg += str(b[count] % 2)
        i += 1
        count += 1
        if i >= len(key_ESK):
            i = 0
    msg_dec = MLEA_dec(msg, key_ESK, key_stkSTEGANO)
    return msg_dec


def generate_msg(n):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-=!@#$%^&*()_+ '
    s = ''
    for i in range(n):
        s += alphabet[random.randint(0, len(alphabet) - 1)]
    return s


# Omsg = open('2kb.txt')
# msg = Omsg.read()


msg = generate_msg(1024 * 8)
msg_bin = str_to_bin(msg)
alg_enc(key, key_shuffle, msg_bin)
msg_b = alg_ext('stego.tiff', key, key_shuffle)
print(bin_to_str(msg_b))
print(msg)

im_org = np.array(Image.open(filename))
im_s = np.array(Image.open('stego.tiff'))

until = 3
from_ = 0


def psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255 ** 2
    return 10 * np.log10(PIXEL_MAX / mse)


def rmse(img1, img2):
    diff = img1 - img2
    diff[diff == 255] = 1
    mse = np.mean(diff ** 2) / 2
    return sqrt(mse)


print(psnr(im_org[:, :, from_:until], im_s[:, :, from_:until]))
print(ssim(im_org[:, :, from_:until], im_s[:, :, from_:until],
           data_range=im_s[:, :, from_:until].max() - im_s[:, :, from_:until].min(), channel_axis=-1))
print(rmse(im_org[:, :, from_:until], im_s[:, :, from_:until]))
