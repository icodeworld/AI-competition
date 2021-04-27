import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from einops import rearrange, repeat
from config import args


class RES_CNN(nn.Module):
    def __init__(self, input_dim=48, kargs=args):
        super(RES_CNN, self).__init__()
        conv_dim = args.cnn_dim
        linear_dim = args.fc_dim
        self.use_convdown_dim = args.use_convdown_dim
        self.drop = nn.Dropout(args.drop_rate)
        self.use_convdown = args.use_convdown_dim
        if args.use_channel_increase:
            self.conv1 = nn.Sequential(nn.Conv2d(input_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                       nn.Conv2d(conv_dim, 2 * conv_dim, 3, stride=2, padding=1), nn.Tanh())
            self.conv2 = nn.Sequential(nn.Conv2d(2 * conv_dim, 2 * conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.Conv2d(2 * conv_dim, 2 * conv_dim, 3, stride=1, padding=1), nn.Tanh())
            self.conv3 = nn.Sequential(nn.Conv2d(2 * conv_dim, 4 * conv_dim, 3, stride=2, padding=1), nn.Tanh())
            self.conv4 = nn.Sequential(nn.Conv2d(4 * conv_dim, 4 * conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.Conv2d(4 * conv_dim, 4 * conv_dim, 3, stride=1, padding=1), nn.Tanh())
            if args.use_convdown_dim:

                self.down = nn.Conv2d(conv_dim * 4, linear_dim, 1, stride=1)
                self.fc = nn.Linear(linear_dim * 27, 1)
            else:
                self.fc = nn.Sequential(nn.Linear(conv_dim * 27 * 4, 1))
        else:

            self.conv1 = nn.Sequential(nn.Conv2d(input_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                       nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh())
            self.conv3 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            self.conv4 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
                                       nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh())
            if args.use_convdown_dim:
                self.down = nn.Conv2d(conv_dim, linear_dim, 1, stride=1)
                self.fc = nn.Linear(linear_dim * 27, 1)
            else:
                self.fc = nn.Sequential(nn.Linear(conv_dim * 27, 1))

    def forward(self, x):
        x1 = self.drop(self.conv1(x))
        x2 = self.drop(self.conv2(x1) + x1)
        x3 = self.drop(self.conv3(x2))
        x4 = self.drop(self.conv4(x3) + x3)
        if self.use_convdown:
            x5 = self.drop(self.down(x4))
            return self.fc(x5.view(x5.size(0), -1))
        else:
            return self.fc(x4.view(x.size(0), -1))


# class RES_CNN(nn.Module):
#     def __init__(self, input_dim=48, kargs=args):
#         super(RES_CNN, self).__init__()
#         conv_dim = args.cnn_dim
#         linear_dim = args.fc_dim
#         self.use_convdown_dim = args.use_convdown_dim
#         self.drop = nn.Dropout(args.drop_rate)
#         self.use_convdown = args.use_convdown_dim
#         if args.use_channel_increase:
#             self.conv1 = nn.Sequential(nn.Conv2d(input_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
#                                        nn.Conv2d(conv_dim, 2 * conv_dim, 3, stride=2, padding=1), nn.Tanh())
#             self.conv2 = nn.Sequential(nn.Conv2d(2 * conv_dim, 2 * conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.Conv2d(2 * conv_dim, 2 * conv_dim, 3, stride=1, padding=1), nn.Tanh())
#             self.conv3 = nn.Sequential(nn.Conv2d(2 * conv_dim, 4 * conv_dim, 3, stride=2, padding=1), nn.Tanh())
#             self.conv4 = nn.Sequential(nn.Conv2d(4 * conv_dim, 4 * conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.Conv2d(4 * conv_dim, 4 * conv_dim, 3, stride=1, padding=1), nn.Tanh())
#             self.fc = nn.Sequential(nn.Linear(conv_dim * 27 * 4, 1))
#         else:
#             self.conv1 = nn.Sequential(nn.Conv2d(input_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
#                                        nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
#             self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1))
#             self.conv3 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
#             self.conv4 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1))
#             # self.conv5 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1), nn.Tanh(),
#             #                            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
#             # self.conv6 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1))
#             self.fc = nn.Sequential(nn.Linear(conv_dim * 27, 1))

#     def forward(self, x):
#         x1 = self.drop(self.conv1(x))
#         x2 = self.drop(self.conv2(x1) + x1)
#         x3 = self.drop(self.conv3(x2))
#         x4 = self.drop(torch.tanh(self.conv4(x3) + x3))
#         # x5 = self.drop(self.conv5(x4))
#         # x6 = self.drop(torch.tanh(self.conv6(x5) + x5))
#         # print("x6 shape", x6.shape)
#         return self.fc(x4.view(x.size(0), -1))