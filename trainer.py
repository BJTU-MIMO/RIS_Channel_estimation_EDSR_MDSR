import os
import math
import numpy as np
from decimal import Decimal
import utility
import torch
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm
from func_forme import downsampling, head_trans_l, head_trans_h, compute_NMSE, ifft_tensor, fft_shrink, calcu_Nmse, \
    reshape, combine_dataset
import scipy.io as sio


Max_abs = 2000


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_sr_height = args.img_sr_height

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):

        img_height = self.img_height            # 256
        img_width = self.img_width              # 32
        img_sr_height = self.img_sr_height      # 64

        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        ################################################################################################################

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

        # train_set1 = data.TensorDataset(train_set)
        # print('H_get', np.max(train_set), np.min(train_set))
        # print('H_get', train_set)
        # print('trainset', torch.max(trainset), torch.min(trainset))
        # train_set1 = data.TensorDataset(train_set)
        # x, y, H, H_get_test = fft_reshape(x_test, img_height, img_width)
        loader_train1111111 = data.DataLoader(train_set, batch_size=10, shuffle=True)
        ################################################################################################################

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, cr in enumerate(loader_train1111111):
            # print(np.shape(cr))
            # print('cr', torch.max(cr), torch.min(cr))   # 问题：输入是零
            print('cr', cr)
            H_lr, H_hr = reshape(cr, img_height, img_width, img_sr_height)
            print('H_lr', H_lr)

            idx_scale = 2
            # lr = downsampling(hr, idx_scale, 64, 32)
            lr_n_fft = head_trans_l(H_lr, img_height, img_width, img_sr_height)

            # hr = hr.cpu().numpy()
            hr_n_fft = head_trans_h(H_hr, img_height, img_width, img_sr_height)
            # hr = torch.from_numpy(hr_n_fft)
            # hr = hr.type(torch.FloatTensor)  # 转Float
            # f'd'sahr = hr.cuda()

            lr = torch.from_numpy(lr_n_fft)
            lr = lr.type(torch.FloatTensor)  # 转Float
            lr = lr.cuda()

            lr = 100 * lr / Max_abs
            hr = 100 * hr / Max_abs

            # print("max:", torch.max(lr), torch.max(hr))
            # print("min:", torch.min(lr), torch.min(hr))

            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()

            sr = self.model(lr, idx_scale)

            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            hr_out = hr * Max_abs / 100
            sr_out = sr * Max_abs / 100

            NMSE = calcu_Nmse(sr_out, hr_out, 64, 32)
            print("NMSE: ", NMSE)

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}--{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:

            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
