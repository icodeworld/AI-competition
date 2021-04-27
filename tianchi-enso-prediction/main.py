import torch
import torch.nn as nn
import model
import train
import json
from torch.utils.data import Dataset
import numpy as np
import netCDF4 as nc
import os
from torch.optim import lr_scheduler
import sys
import random
from torchsummary import summary
from config import args

CMIP_INPUT = "../tcdata/enso_round1_test_20210201/CMIP_train.nc"
CMIP_LABEL = "../tcdata/enso_round1_test_20210201/CMIP_label.nc"
SODA_INPUT = "../tcdata/enso_round1_test_20210201/SODA_train.nc"
SODA_LABEL = "../tcdata/enso_round1_test_20210201/SODA_label.nc"
CMIP_INPUT_48 = "../data_48/CMIP_train_48.nc"
CMIP_LABEL_48 = "../data_48/CMIP_label_48.nc"
SODA_INPUT_48 = "../data_48/SODA_train_48.nc"
SODA_LABEL_48 = "../data_48/SODA_label_48.nc"

seed = 42
NFLAG = 1
key_to_value = {True: '1', False: '0'}
# serial_number = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = serial_number


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.deterministic = True


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def SstdataLoader(path1, path2, end_mon, lead_mon, ntype, args):
    file1 = nc.Dataset(path1)
    file2 = nc.Dataset(path2)
    input_mon = args.in_monthes
    old_start_monthes = 12 - input_mon
    offset_mon = end_mon - 12
    start_monthes = old_start_monthes + offset_mon
    if args.use_wind:
        num_factors = 4
    else:
        num_factors = 1
    if ntype == 'CMIP':
        if args.use_align:
            inputs = np.zeros((4613, input_mon * num_factors, 24, 72))
        else:
            inputs = np.zeros((4645, input_mon * num_factors, 24, 72))
    elif ntype == 'SODA':
        if args.use_align:
            inputs = np.zeros((99, input_mon * num_factors, 24, 72))
        else:
            inputs = np.zeros((100, input_mon * num_factors, 24, 72))

    if args.use_wind:
        inputs[:, 0:input_mon, :, :] = file1.variables['sst'][:, start_monthes:(start_monthes + input_mon), :, :]
        inputs[:, input_mon:2 * input_mon, :, :] = file1.variables['t300'][:, start_monthes:(start_monthes + input_mon), :, :]
        inputs[:, 2 * input_mon:3 * input_mon, :, :] = file1.variables['ua'][:, start_monthes:(start_monthes + input_mon), :, :]
        inputs[:, 3 * input_mon:4 * input_mon, :, :] = file1.variables['va'][:, start_monthes:(start_monthes + input_mon), :, :]
    else:
        inputs[:, 0:input_mon, :, :] = file1.variables['sst'][:, start_monthes:(start_monthes + input_mon), :, :]
        #inputs[:, input_mon:2 * input_mon, :, :] = file1.variables['t300'][:, start_monthes:(start_monthes + input_mon), :, :]
    inputs[np.isnan(inputs)] = 0
    labels = file2.variables['nino'][:, (11 + lead_mon + offset_mon):(12 + lead_mon + offset_mon)]
    if args.augmentation == 'none':
        return inputs, labels
    elif args.augmentation == 'class':
        # print(inputs)
        inputs[(inputs > -30) & (inputs <= -3)] = -3
        inputs[(inputs > 3) & (inputs <= 30)] = 3
        return inputs, labels
    elif args.augmentation == 'class2':
        inputs[(inputs > -30) & (inputs <= -3)] = -3
        inputs[(inputs > -3) & (inputs <= -2.5)] = -2.5
        inputs[(inputs > -2.5) & (inputs <= -2)] = -2
        inputs[(inputs > -2) & (inputs <= -1.5)] = -1.5
        inputs[(inputs > -1.5) & (inputs < -1)] = -1
        inputs[(inputs > -1) & (inputs <= -0.5)] = -0.5
        inputs[(inputs > -0.5) & (inputs <= 0)] = 0
        inputs[(inputs > 0) & (inputs <= 0.5)] = 0.5
        inputs[(inputs > 0.5) & (inputs <= 1)] = 1
        inputs[(inputs > 1) & (inputs <= 1.5)] = 1.5
        inputs[(inputs > 1.5) & (inputs <= 2)] = 2
        inputs[(inputs > 2) & (inputs <= 2.5)] = 2.5
        inputs[(inputs > 2.5) & (inputs <= 3)] = 3
        inputs[(inputs > 3) & (inputs <= 30)] = 3.5
        return inputs, labels


class Sstdataset(Dataset):
    def __init__(self, path1, path2, end_mon=12, lead_mon=1, ntype='None', kargs=args):

        self.data = SstdataLoader(path1, path2, end_mon, lead_mon, ntype, args)
        self.inputs, self.lables = self.data

    def __len__(self):
        return len(self.inputs[:, 0, 0, 0])

    def __getitem__(self, indx):
        ##getitem method
        self.x = torch.from_numpy(self.inputs[indx, ...]).float().to('cpu')
        # print("shape", self.x.shape)
        self.y = torch.from_numpy(self.lables[indx, ...]).float().to('cpu')
        return self.x, self.y


def main(end_months, lead_months, epoch_epoch, args):
    global NFLAG
    if args.use_align:
        train_set = Sstdataset(CMIP_INPUT_48, CMIP_LABEL_48, end_months, lead_months, ntype='CMIP', kargs=args)
        valid_set = Sstdataset(SODA_INPUT_48, SODA_LABEL_48, end_months, lead_months, ntype='SODA', kargs=args)

    else:
        train_set = Sstdataset(CMIP_INPUT, CMIP_LABEL, end_months, lead_months, ntype='CMIP', kargs=args)
        valid_set = Sstdataset(SODA_INPUT, SODA_LABEL, end_months, lead_months, ntype='SODA', kargs=args)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchtrain, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False)
    if args.use_wind:
        num_factors = 4
    else:
        num_factors = 1
    input_channel = args.in_monthes * num_factors
    net = model.RES_CNN(input_dim=input_channel, kargs=args).to('cpu')
    # if NFLAG == 1:
    #     summary(net, input_size=(input_channel, 24, 72))
    #     NFLAG = 0
    optimizer1 = torch.optim.Adam(net.parameters(), lr=args.lr1)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=7, gamma=args.weight_decay)
    if args.lossf == 'mse':
        criterion = nn.MSELoss()
    elif args.lossf == 'mae':
        criterion = nn.L1Loss()
    else:
        pass
    if args.use_transfer:
        # net.load
        pass
    net = train.train(net,
                      train_loader,
                      valid_loader,
                      args,
                      args.transfer_epoch,
                      optimizer1,
                      criterion,
                      exp_lr_scheduler1,
                      end_months,
                      lead_months,
                      epoch_epoch,
                      mode='train_val')
    optimizer2 = torch.optim.Adam(net.parameters(), lr=args.lr2)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=2, gamma=args.weight_decay)
    net = train.train(net,
                      valid_loader,
                      valid_loader,
                      args,
                      args.train_epoch,
                      optimizer2,
                      criterion,
                      exp_lr_scheduler2,
                      end_months,
                      lead_months,
                      epoch_epoch,
                      mode='after')

    # return train.test(net, criterion, test_loader, end_months, lead_months)


seed_everything(seed)
dir_name = 'rescnn_'

for times in range(0, 1):
    # drop_name = 'Drop_' + str(args.drop_rate)
    current_dir_name = dir_name + key_to_value[args.use_wind] + '-' + str(args.in_monthes) + '-' + key_to_value[
        args.use_align] + args.augmentation + args.lossf + key_to_value[args.use_channel_increase] + str(args.cnn_dim) + str(
            args.fc_dim) + key_to_value[args.use_convdown_dim] + str(args.drop_rate) + key_to_value[args.use_transfer] + str(
                args.lr1) + '-' + str(args.lr2) + '-' + str(args.transfer_epoch) + '-' + str(args.train_epoch) + '-' + str(
                    args.seed)
    if not os.path.exists(current_dir_name):
        os.makedirs(current_dir_name)
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), current_dir_name)))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('log.txt')
    for lead in range(1, 25):
        if args.use_align:
            end_month = range(12, 24)
        else:
            end_month = range(12, 13)
        for tar in end_month:
            main(tar, lead, times, args)
