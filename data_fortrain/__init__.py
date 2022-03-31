import scipy.io as sio
from func_forme import combine_dataset
import torch.utils.data as data
import numpy as np
import torch


class Data:
    def __init__(self, args):
        self.loader_train = None
        mat_train_one = sio.loadmat('data_fortrain/befortrain.mat')
        # mat_train_two = sio.loadmat('data_fortrain/train_two.mat')
        # mat_train_three = sio.loadmat('data_fortrain/train_three.mat')
        # mat_train_four = sio.loadmat('data_fortrain/train_four.mat')

        x_train_one = mat_train_one['H_combine']
        # x_train_two = mat_train_two['H_combine']
        # x_train_three = mat_train_three['H_combine']
        # x_train_four = mat_train_four['H_combine']

        mat_test = sio.loadmat('data_fortrain/test_one.mat')
        x_test = mat_test['H_combine']

        # trainset = combine_dataset(x_train_one, x_train_two, x_train_three, x_train_four)
        train_set = combine_dataset(x_train_one)
        print('H_get', np.max(train_set), np.min(train_set))
        # print('H_get', train_set)
        # print('trainset', torch.max(trainset), torch.min(trainset))
        # train_set1 = data.TensorDataset(train_set)
        # x, y, H, H_get_test = fft_reshape(x_test, img_height, img_width)
        self.loader_train = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_set = combine_dataset(x_test)
        self.loader_test = data.DataLoader(test_set, batch_size=1, shuffle=False)
