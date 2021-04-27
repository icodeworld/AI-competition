import torch
import torch.nn as nn
import model
from torch.utils.data import Dataset
import numpy as np
import netCDF4 as nc
import os
from config import args
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
# 初始保存所有文件名方便之后按照顺序统一保存
import glob, zipfile
from config import args

# path = glob.glob('tcdata/enso_round1_test_20210201/*')
my_path = glob.glob('tcdata/enso_final_test_data_B/*')
TOTAL_PREDICTION = np.zeros((len(my_path), 24))
# TOTAL_PREDICTION = np.zeros((2, 24))


def fuction_zip():
    z = zipfile.ZipFile('result.zip', 'w')
    testdir = 'result'
    if os.path.isdir(testdir):
        for d in os.listdir(testdir):
            z.write(testdir + os.sep + d)
        z.close()
    else:
        print('None')


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))


# TODO:将所有测试数据的input放入一个文件中，格式为npy,维度（batch，12，24，72，4）
# label放入一个文件，格式为（batch，12）
def process_data(sep_file_path, lead_mon, end_mon, args):
    # file:
    file1 = np.load(sep_file_path).reshape(1, 12, 24, 72, 4)
    file1[np.isnan(file1)] = 0
    # data preprocess
    input_mon = args.in_monthes
    start_monthes = 12 - input_mon
    if args.use_wind:
        num_factors = 4
    else:
        num_factors = 1
    inputs = np.zeros((1, input_mon * num_factors, 24, 72))
    if args.use_wind:
        inputs[:, 0:input_mon, :, :] = file1[:, start_monthes:12, :, :, 0]
        inputs[:, input_mon:2 * input_mon, :, :] = file1[:, start_monthes:12, :, :, 1]
        inputs[:, 2 * input_mon:3 * input_mon, :, :] = file1[:, start_monthes:12, :, :, 2]
        inputs[:, 3 * input_mon:4 * input_mon, :, :] = file1[:, start_monthes:12, :, :, 3]
    else:
        inputs[:, 0:input_mon, :, :] = file1[:, start_monthes:12, :, :, 0]
        #inputs[:, input_mon:2 * input_mon, :, :] = file1[:, start_monthes:12, :, :, 1]
    inputs = torch.from_numpy(inputs).float()
    return inputs


def save_and_score(prediction):
    # save prediction file
    file_path = glob.glob('tcdata/enso_final_test_data_B/*')
    if not os.path.isdir('result'):
        os.mkdir('result')
    #  save result
    for i in range(prediction.shape[0]):
        np.save("result/" + file_path[i][-20:], prediction[i, :])


    # 计算分数
def test(net, index, true_input, lead_month, end_mon, args):
    with torch.no_grad():
        output = net(true_input)  # [batch]
        # print("output", output[:, 0].detach().numpy())
        TOTAL_PREDICTION[index, lead_month - 1] = output[0, 0].detach().numpy()  #.cpu().detach().numpy()


key_to_value = {True: '1', False: '0'}

# path = glob.glob('my_test_file/*')
experient_name = 'rescnn_' + key_to_value[args.use_wind] + '-' + str(args.in_monthes) + '-' + key_to_value[
    args.use_align] + args.augmentation + args.lossf + key_to_value[args.use_channel_increase] + str(args.cnn_dim) + str(
        args.fc_dim) + key_to_value[args.use_convdown_dim] + str(args.drop_rate) + key_to_value[args.use_transfer] + str(
            args.lr1) + '-' + str(args.lr2) + '-' + str(args.transfer_epoch) + '-' + str(args.train_epoch) + '-' + str(
                args.seed)
for leads in range(1, 25):
    if args.use_wind:
        num_factors = 4
    else:
        num_factors = 1
    if args.soda:
        model_dir_path = '/after'
    else:
        model_dir_path = '/train_val'
    input_channel = args.in_monthes * num_factors
    net = model.RES_CNN(input_dim=input_channel, kargs=args)
    for year_index, p in enumerate(my_path):
        # print("path", year_index, p[-6:-4])
        end_mon = int(p[-6:-4]) % 12 + 12
        # train_val
        if not args.use_align:
            end_mon = 12
        inputs = process_data(p, leads, end_mon, args)
        parameter = torch.load(experient_name + model_dir_path + '/model_0/transfer_T' + str(end_mon) + 'L' + str(leads),
                               map_location=lambda storage, loc: storage)
        net.load_state_dict(parameter)
        net = net.eval()
        test(net, year_index, inputs, leads, end_mon, args)
print("name", experient_name + model_dir_path)
save_and_score(TOTAL_PREDICTION)
fuction_zip()
# now TOTAL_PREDICTION [104, 24],依次根据文件名保存即可
print('done')
