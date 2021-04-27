import argparse

parser = argparse.ArgumentParser(description='RES_CNN for ENSO')

# # data selection
parser.add_argument('--use_wind', action='store_true')
parser.add_argument('--in_monthes', type=int, default=3)
parser.add_argument('--use_align', action='store_true')
parser.add_argument('--augmentation', type=str, default='none', help='remove, gaosi,other')

# loss function
parser.add_argument('--lossf', type=str, default='mse', help='mse,mae,other')

# model param
# inputdim is cal by the data selction
parser.add_argument('--use_channel_increase', action='store_true')
parser.add_argument('--cnn_dim', type=int, default=30, help='32,64,128,196')
parser.add_argument('--fc_dim', type=int, default=5, help='32,64,128,196')
parser.add_argument('--use_convdown_dim', action='store_true')
parser.add_argument('--drop_rate', type=float, default=0, help='dropout_rate')
parser.add_argument('--seed', type=int, default=42, help='dropout_rate')
# train ways
parser.add_argument('--soda', action='store_true', help='use the SODA model params as the current model')
parser.add_argument('--use_transfer', action='store_true', help='use the last model params as the current model')

# not change
parser.add_argument("--transfer_epoch", type=int, default=5, help='number of epochs to train')  # 10
parser.add_argument("--train_epoch", type=int, default=20, help='number of epochs to train')  # 50
parser.add_argument("--batchtrain", type=int, default=128, help='the batch for train')  # 128
parser.add_argument("--lr1", type=float, default=0.001, help='cmip learning rate')
parser.add_argument("--lr2", type=float, default=0.00009, help='soda learning rate')
parser.add_argument("--weight_decay", type=float, default=0.9, help='weight decay')
parser.add_argument("--train_show_epoch", type=int, default=1, help='show result every epoch')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
