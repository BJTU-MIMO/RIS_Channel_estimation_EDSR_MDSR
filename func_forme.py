import numpy as np
import math
from scipy.fftpack import fft2, ifft2, fft, ifft
import torch


def reshape(x_train_one, img_height, img_width, img_sr_height):
    # img_height     256
    # img_width      32
    # img_sr_height  64
    y = len(x_train_one)
    H_hr = np.zeros([y, 1, img_height, 2*img_width], dtype=complex)
    H_lr = np.zeros([y, 1, img_sr_height, 2*img_width], dtype=complex)

    for i in range(y):
        H_hr[i, 0, 0:img_height, 0:2*img_width] = x_train_one[i, 0, 0:img_height, 0:2*img_width]
        H_lr[i, 0, 0:img_sr_height, 0:2*img_width] = x_train_one[i, 0, img_height:(img_height+img_sr_height), 0:2*img_width]

    return H_lr, H_hr


def fft_reshape(H_shape, img_height, img_width):
    x = len(H_shape)    # 2048
    y = len(H_shape[0])    # n
    H_reshape = np.zeros([y, img_width, img_height], dtype=complex)
    H_reshape_T = np.zeros([y, img_height, img_width], dtype=complex)
    H_real = np.zeros([y, img_height, img_width])
    H_imag = np.zeros([y, img_height, img_width])
    H_get = np.zeros([y, img_height, img_width*2])
    for i in range(y):
        H_reshape[i, :] = np.reshape(H_shape[:, i], (img_width, img_height))
        H_reshape_T[i, :] = H_reshape[i, :].T
        H_real[i, :] = H_reshape_T[i, :].real
        H_imag[i, :] = H_reshape_T[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return x, y, H_reshape_T, H_get


def combine_dataset(x_train_one):
    # , x_train_two, x_train_three, x_train_four
    y = 100
    img_width = 32
    img_sr_height = 64
    img_height = 256
    heigh_together = img_height + img_sr_height

    H_set = np.zeros([y, heigh_together, img_width], dtype=complex)

    # print('H_get', np.max(x_train_one), np.min(x_train_one))  # 没问题
    # print(np.shape(x_train_one))

    for i in range(y):
        H_set[i, 0:heigh_together, 0:img_width] = x_train_one[i, 0:heigh_together, 0:img_width]
        # H_set[i+y, 0:heigh_together, 0:img_width] = x_train_two[i, 0:heigh_together, 0:img_width]
        # H_set[i+2*y, 0:heigh_together, 0:img_width] = x_train_three[i, 0:heigh_together, 0:img_width]
        # H_set[i+3*y, 0:heigh_together, 0:img_width] = x_train_four[i, 0:heigh_together, 0:img_width]

    # print('H_get', np.max(H_set), np.min(H_set))

    H_real = np.zeros([y, heigh_together, img_width])
    H_imag = np.zeros([y, heigh_together, img_width])
    H_get = np.zeros([y, heigh_together, img_width * 2])
    H_get1 = np.zeros([y, 1, heigh_together, img_width * 2])

    for i in range(y):
        H_real[i, 0:heigh_together, 0:img_width] = H_set[i, 0:heigh_together, 0:img_width].real
        H_imag[i, 0:heigh_together, 0:img_width] = H_set[i, 0:heigh_together, 0:img_width].imag
        H_get[i, 0:heigh_together, 0:2*img_width] = np.hstack((H_real[i, 0:heigh_together, 0:img_width], H_imag[i, 0:heigh_together, 0:img_width]))
        H_get1[i, 0, 0:heigh_together, 0:2 * img_width] = H_get[i, 0:heigh_together, 0:2 * img_width]

    # torch_data = torch.from_numpy(H_get1)

    # print('H_get', np.max(H_get), np.min(H_get))

    # H_get1 = H_get1.astype(np.float64)
    # print('H_get', np.max(H_get), np.min(H_get))
    return H_get1


def downsampling(H, idx_scale, img_height, img_width):
    y = len(H[0])
    H2 = np.zeros([y, 1, img_width, img_width])
    for i in range(y):
        H2[i, 0, 0:img_width, 0:img_width] = H[i, 0, 0:idx_scale:64, 0:idx_scale:64]
    return H2


def head_trans_l(lr, img_height, img_width, img_sr_height):
    NN = len(lr)
    print('H-lr', lr[0, 10:30])
    H = fft_shrink(lr, img_sr_height, img_width)     # n 1 64 32
    noise = add_noise_improve(H, 80, 80.5, img_sr_height, img_width)
    H_n = noise + H
    print('H', H[0, 10:30])
    print('noise', noise[0, 10:30])
    print('H_n', H_n[0, 10:30])
    H2 = np.zeros([NN, img_sr_height, img_width], dtype=complex)
    for i_num in range(NN):
        H2[i_num, :, :] = fft2(H_n[i_num, 0, :, :])

    # print('H2', H2[0, 10:30])
    H_n_fft_stack1 = real_imag_stack(H2, img_sr_height, img_width)
    return H_n_fft_stack1


def head_trans_h(lr, img_height, img_width, img_sr_height):
    NN = len(lr)
    H = fft_shrink(lr, 64, 32)

    H2 = np.zeros([NN, 64, 32], dtype=complex)
    H3 = np.zeros([NN, 64, 32], dtype=complex)
    for i_num in range(NN):
        H3[i_num, 0:64, 0:32] = H[i_num, 0, 0:64, 0:32]
        H2[i_num, :, :] = fft2(H3[i_num, :, :])

    H_n_fft_stack1 = real_imag_stack(H2, 64, 32)
    return H_n_fft_stack1


def fft_shrink(H_shape, img_height, img_width):
    y = len(H_shape)
    H_real = np.zeros([y, 1, img_height, img_width], dtype=complex)
    H_imag = np.zeros([y, 1, img_height, img_width], dtype=complex)
    for i in range(y):
        H_real[i, 0, 0:img_height, 0:img_width] = H_real[i, 0, :, :] + H_shape[i, 0, 0:img_height, 0:img_width]
        H_imag[i, 0, 0:img_height, 0:img_width] = H_real[i, 0, :, :] + 1j*H_shape[i, 0, 0:img_height, img_width:img_height]

    return H_imag


def add_noise_improve(input, SNRlow, SNRhign, img_sr_height, img_width):
    NN = len(input)
    # print('NN:', NN)

    noise = np.zeros([NN, 1, img_sr_height, img_width], dtype=complex) # n 1 64 32
    SNR_divide = np.random.uniform(SNRlow, SNRhign, size=[NN])
    stdn = np.random.uniform(0.01, 0.02, size=[NN])
    for nx in range(NN):
        # print(np.shape(input[nx, 0, :, :]))
        noise[nx, 0, :, :] = add_noise1(input[nx, 0, :, :], SNR_divide[nx], img_sr_height, img_width)

    # noise = add_noise1(input, SNR_divide, img_sr_height, img_width)
    return noise


def add_noise1(input, SNR, img_sr_height, img_width):
    NN = len(input)
    # print(np.shape(input))
    input11 = np.zeros([img_sr_height, img_width], dtype=complex)
    input11[0:img_sr_height, 0:img_width] = input[0:img_sr_height, 0:img_width]
    x_test_realc = np.reshape(input11, (1, -1))
    # print(np.shape(x_test_realc))
    power = np.sum(abs(x_test_realc) ** 2, axis=1)
    power_level = power
    power_level_1 = power_level/(img_sr_height * img_width)
    SNR_level = 10**(SNR/10)
    noise_level = power_level_1/SNR_level
    # print(np.shape(noise_level))
    n_l = math.sqrt(noise_level)
    Noise_map = np.zeros([1, 1, img_sr_height, img_width], dtype=complex)
    Noise_map_real = np.zeros([1, 1, img_sr_height, img_width], dtype=complex)
    Noise_map_imag = np.zeros([1, 1, img_sr_height, img_width], dtype=complex)
    Noise_map_real = Noise_map_real + n_l * math.sqrt(1 / 2) * np.random.randn(NN, 1, img_sr_height, img_width)
    Noise_map_imag = Noise_map_imag + 1j * n_l * math.sqrt(1 / 2) * np.random.randn(NN, 1, img_sr_height, img_width)

    Noise_map[0, 0, :, :] = Noise_map_real[0, 0, :, :] + Noise_map_imag[0, 0, :, :]
    # print('Noise_map', np.shape(Noise_map))

    return Noise_map


def real_imag_stack(H, img_height, img_width):
    x = len(H)    # n
    H_real = np.zeros([x, 1, img_height, img_width])
    H_imag = np.zeros([x, 1, img_height, img_width])
    H_out = np.zeros([x, 1, img_height, 2*img_width])
    for i in range(x):
        H_real[i, 0, :, :] = H[i, :, :].real
        H_imag[i, 0, :, :] = H[i, :, :].imag
        H_out[i, 0, :, :] = np.hstack((H_real[i, 0, :, :], H_imag[i, 0, :, :]))
    return H_out


def compute_NMSE(H, H_pre):
    H1 = np.reshape(H, (len(H), -1))
    H_pre1 = np.reshape(H_pre, (len(H_pre), -1))
    power = np.sum(abs(H_pre1) ** 2, axis=1)
    mse = np.sum(abs(H1 - H_pre1) ** 2, axis=1)
    NMSE = 10 * math.log10(np.mean(mse / power))
    return NMSE


def ifft_tensor(H_shape):
    x = len(H_shape)    # n
    H_reshape_fft = np.zeros([x, 64, 32], dtype=complex)
    H_real = np.zeros([x, 64, 32])
    H_imag = np.zeros([x, 64, 32])
    H_get = np.zeros([x, 64, 64])
    for i in range(x):
        H_reshape_fft[i, :, :] = ifft2(H_shape[i, :, :])
        H_real[i, :] = H_reshape_fft[i, :].real
        H_imag[i, :] = H_reshape_fft[i, :].imag
        H_get[i, :] = np.hstack((H_real[i, :], H_imag[i, :]))
    return H_reshape_fft, H_get


def calcu_Nmse(sr, hr, img_height, img_width):
    sr = sr.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    # print(type(sr))
    sr_i = np.zeros([1, 64, 64], dtype=complex)  # 64 X 64
    sr_i = sr + sr_i

    hr_i = np.zeros([1, 64, 64], dtype=complex)  # 64 X 64
    hr_i = hr + hr_i
    sr_fft = fft_shrink(sr_i, img_height, img_width)
    hr_fft = fft_shrink(hr_i, img_height, img_width)
    srr, H_fft_pre_last2 = ifft_tensor(sr_fft)
    hrr, H_fft_pre_last2 = ifft_tensor(hr_fft)
    NMSE = compute_NMSE(srr, hrr)
    return NMSE
