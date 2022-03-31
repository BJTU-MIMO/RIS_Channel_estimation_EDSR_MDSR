import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
import data_fortrain
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
import os
import utility
from option import args
import model
import loss
from trainer import Trainer


os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDE_VISIBLE_DIVICES"] = "0"

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


if checkpoint.ok:
    model = model.Model(args, checkpoint)
    # print(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    loader = data_fortrain.Data(args)
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()

    checkpoint.done()
