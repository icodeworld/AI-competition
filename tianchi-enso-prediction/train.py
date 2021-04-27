import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import netCDF4 as nc
import os
import json
import numpy as np
import copy
# import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(net,
          train_data,
          valid_data,
          args,
          num_epochs,
          optimizer,
          criterion,
          scheduler,
          end_months,
          lead_months,
          epoch_epoch,
          mode='before'):  # criterion

    best_loss = 1e10
    save_el = []
    py = []
    net = net.train()
    for epoch in range(0, num_epochs):
        train_loss = 0
        valid_loss = 0
        prev_time = time.time()
        # FIXME:
        for seq, seq_target in train_data:
            seq = Variable(
                seq.to('cpu')  #.cuda()  # [N, S, H, W]
                ,
                requires_grad=True)  # [N, S, H, W]
            #seq_target = Variable(seq_target.cuda(), requires_grad=True)  # [N, S, H, W]
            seq_target = Variable(seq_target.to('cpu'), requires_grad=True)  # [N, S, H, W]
            # forward
            optimizer.zero_grad()
            output = net(seq)  # [N, S, H, W]
            loss = criterion(output, seq_target)
            train_loss += float(loss.item())
            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()
        #FIXME:
        train_loss = train_loss / len(train_data)
        save_el.append([epoch, train_loss])
        # test
        net = net.eval()
        for valid_seq, valid_target in valid_data:
            with torch.no_grad():
                #valid_seq = Variable(valid_seq.cuda(), requires_grad=False)  # [N, S, H, W]
                #valid_target = Variable(valid_target.cuda(), requires_grad=False)  # [N, S, H, W]
                valid_seq = Variable(valid_seq.to('cpu'), requires_grad=False)  # [N, S, H, W]
                valid_target = Variable(valid_target.to('cpu'), requires_grad=False)  # [N, S, H, W]
                valid_output = net(valid_seq)  # [N, S, H, W]
                loss = criterion(valid_output, valid_target)
                valid_loss += loss.item()
        valid_loss = valid_loss / len(valid_data)
        py.append(valid_loss)
        if best_loss >= valid_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(net.state_dict())
            # best_output = valid_output

        if (epoch + 1) == 1 or (epoch + 1) % args.train_show_epoch == 0:
            print("epoch: %d, t_loss: %.4f,valid_loss: %.4f,best_loss: %.4f, T: %d, L: %d" %
                  (epoch + 1, train_loss, valid_loss, best_loss, end_months, lead_months))
    epp = np.array(save_el)[:, 0]
    trl = np.array(save_el)[:, 1]
    # store the result each time
    if not os.path.isdir(mode):
        os.mkdir(mode)
    if not os.path.isdir(mode + '/loss_' + str(epoch_epoch)):
        os.mkdir(mode + '/loss_' + str(epoch_epoch))
    if not os.path.isdir(mode + '/model_' + str(epoch_epoch)):
        os.mkdir(mode + '/model_' + str(epoch_epoch))
    if not os.path.isdir(mode + '/best_model_' + str(epoch_epoch)):
        os.mkdir(mode + '/best_model_' + str(epoch_epoch))
    if not os.path.isdir(mode + '/output_' + str(epoch_epoch)):
        os.mkdir(mode + '/output_' + str(epoch_epoch))

    np.savez(mode + "/loss_" + str(epoch_epoch) + "/loss" + "_T" + str(end_months) + "L" + str(lead_months) + ".npz",
             epoch=epp,
             tr=trl,
             te=py)
    torch.save(best_model_wts,
               mode + "/best_model_" + str(epoch_epoch) + "/transfer" + "_T" + str(end_months) + "L" + str(lead_months))
    torch.save(net.state_dict(),
               mode + "/model_" + str(epoch_epoch) + "/transfer" + "_T" + str(end_months) + "L" + str(lead_months))
    # net.load_state_dict(best_model_wts)

    return net


def test(net, criterion, test_data, target_month, lead_month):
    # logger.info('test start.  ')
    test_loss = 0
    net = net.eval()
    # prev_time = time.time()

    for seq, seq_target in test_data:
        with torch.no_grad():
            #seq = Variable(seq.cuda(), requires_grad=False)  # [N, S, H, W]
            #seq_target = Variable(seq_target.cuda(), requires_grad=False)  # [N, S, H, W]
            seq = Variable(seq, requires_grad=False)  # [N, S, H, W]
            seq_target = Variable(seq_target, requires_grad=False)  # [N, S, H, W]
            output = net(seq)  # [N, S, H, W]
            loss = criterion(output, seq_target)
            test_loss += loss.item()
    test_loss = test_loss / len(test_data)
    print("test_loss: %.4f, T: %d, L: %d" % (test_loss, target_month, lead_month))
    return test_loss
    # return round(test_loss, 4)
