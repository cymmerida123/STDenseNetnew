# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by ct
** Name: demo
** Date: 1/21/18
** BSD license
********************************************/
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from einops import rearrange

sys.path.append('../')
# from dataloader.milano import load_data
from models.DenseNet import DenseNet
from dataloader.datasetSW import MilanSlidingWindowDataset

torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=20)
parse.add_argument('-width', type=int, default=20)
parse.add_argument('-traffic', type=str, default='sms')
# parse.add_argument('-close_size', type=int, default=3)
# parse.add_argument('-period_size', type=int, default=3)
# parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-input_len', type=int, default=24)
parse.add_argument('-gis_num', type=int, default=0)
parse.add_argument('-window_size', type=int, default=10)
parse.add_argument('-test_size', type=int, default=24 * 7)
parse.add_argument('-nb_flow', type=int, default=24)
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=False)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-rows', nargs='+', type=int, default=[5, 15])
parse.add_argument('-cols', nargs='+', type=int, default=[5, 15])
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.001)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=10, help='epochs')
parse.add_argument('-drop_rate', type=float, default=0.2, help='drop out rate')
parse.add_argument('-test_row', type=int, default=5, help='test row')
parse.add_argument('-test_col', type=int, default=4, help='test col')

parse.add_argument('-save_dir', type=str, default='./results')

opt = parse.parse_args()
# print(opt)
# opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
opt.model_filename = '{}/model={}-loss={}-lr={}-input_len={}'.format(opt.save_dir,
                                          'densenet',
                                          opt.loss, opt.lr, opt.input_len)


# print('Saving to ' + opt.model_filename)


def log(fname, s):
    # if not os.path.isdir(os.path.dirname(fname)):
    #     os.system("mkdir -p " + os.path.dirname(fname))
    # fname = opt.save_dir + '/' + fname
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    if float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(data_type='train'):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader

    # if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
    #     for idx, (c, p, t, target) in enumerate(data):
    #         optimizer.zero_grad()
    #         model.zero_grad()
    #         input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
    #         target_var = Variable(target.float(), requires_grad=False).cuda()

    #         pred = model(input_var)
    #         loss = criterion(pred, target_var)
    #         total_loss += loss.item()
    #         if data_type == 'train':
    #             loss.backward()
    #             optimizer.step()
    # if (opt.gis_num > 0) & (opt.input_len > 0):
    #     for idx, (c, p, target) in enumerate(data):
    #         optimizer.zero_grad()
    #         model.zero_grad()
    #         input_var = [Variable(_.float()).cuda() for _ in [c, p]]
    #         target_var = Variable(target.float(), requires_grad=False).cuda()

    #         pred = model(input_var)
    #         loss = criterion(pred, target_var)
    #         total_loss += loss.item()
    #         if data_type == 'train':
    #             loss.backward()
    #             optimizer.step()
    # elif opt.gis_num > 0:
    #     for idx, (c, target) in enumerate(data):
    #         optimizer.zero_grad()
    #         model.zero_grad()
    #         x = [Variable(c.float()).cuda()]
    #         y = Variable(target.float(), requires_grad=False).cuda()

    #         pred = model(x)
    #         loss = criterion(pred, y)
    #         total_loss += loss.item()
    #         if data_type == 'train':
    #             loss.backward()
    #             optimizer.step()
    for idx, (c, p, target) in enumerate(data):
        optimizer.zero_grad()
        model.zero_grad()
        input_var = [Variable(_.float()).cuda() for _ in [c, p]]
        target_var = Variable(target.float(), requires_grad=False).cuda()

        pred = model(input_var)
        loss = criterion(pred, target_var)
        total_loss += loss.item()
        if data_type == 'train':
            loss.backward()
            optimizer.step()

    return total_loss


def train():
    # os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 100
    train_loss, valid_loss = [], []
    early_stop_num = 10
    early_stop = 0
    for i in range(opt.epoch_size):
        train_loss.append(train_epoch('train'))
        valid_loss.append(train_epoch('valid'))
        scheduler.step()

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')
            early_stop = 0
        else:
            early_stop += 1

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}').format((i + 1), opt.epoch_size,
                                                         train_loss[-1],
                                                         valid_loss[-1],
                                                         best_valid_loss)
        print(log_string)
        log(opt.model_filename + '.log', log_string)
        if early_stop >= early_stop_num:
            print("Early stop...")
            break


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    best_model = torch.load(opt.model_filename + '.model').get('model')

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    # if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
    #     for idx, (c, p, t, target) in enumerate(data):
    #         input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
    #         target_var = Variable(target.float(), requires_grad=False).cuda()
    #         pred = best_model(input_var)
    #         predictions.append(pred.data.cpu().numpy())
    #         ground_truth.append(target.numpy())
    #         loss.append(criterion(pred, target_var).item())
    # if (opt.gis_num > 0) & (opt.input_len > 0):
    #     for idx, (c, p, target) in enumerate(data):
    #         input_var = [Variable(_.float()).cuda() for _ in [c, p]]
    #         target_var = Variable(target.float(), requires_grad=False).cuda()
    #         pred = best_model(input_var)
    #         predictions.append(pred.data.cpu().numpy())
    #         ground_truth.append(target.numpy())
    #         loss.append(criterion(pred, target_var).item())
    # elif opt.gis_num > 0:
    #     for idx, (c, target) in enumerate(data):
    #         input_var = [Variable(c.float()).cuda()]
    #         target_var = Variable(target.float(), requires_grad=False).cuda()
    #         pred = best_model(input_var)
    #         predictions.append(pred.data.cpu().numpy())
    #         ground_truth.append(target.numpy())
    #         loss.append(criterion(pred, target_var).item())

    for idx, (c, p, target) in enumerate(data):
        input_var = [Variable(_.float()).cuda() for _ in [c, p]]
        target_var = Variable(target.float(), requires_grad=False).cuda()
        pred = best_model(input_var)
        predictions.append(pred.data.cpu().numpy())
        ground_truth.append(target.numpy())
        loss.append(criterion(pred, target_var).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    B,T,H,W = final_predict.shape
    final_predict = final_predict.reshape(B//100,100,T,H,W)
    final_predict = final_predict.transpose(0,2,1,3,4)
    ground_truth = ground_truth.reshape(B//100,100,T,H,W)
    ground_truth = ground_truth.transpose(0,2,1,3,4)
    final_predict_rec = np.zeros([B//100,T,100,100])
    ground_truth_rec = np.zeros([B//100,T,100,100])
    for i in range(100):
        col = i//10
        row = i%10
        final_predict_rec[:,:,col*10:(col+1)*10,row*10:(row+1)*10] = final_predict[:,:,i,:,:]
        ground_truth_rec[:,:,col*10:(col+1)*10,row*10:(row+1)*10] = ground_truth[:,:,i,:,:]

    final_predict_rec = mmn.inverse_transform(final_predict_rec.reshape(B//100*T,100*100))
    ground_truth_rec = mmn.inverse_transform(ground_truth_rec.reshape(B//100*T,100*100))

    rmse = metrics.mean_squared_error(final_predict_rec, ground_truth_rec) ** 0.5
    print(test_type + ' RMSE:{:0.5f}'.format(np.mean(rmse)))
    # print(len(model['train_loss']))

    if opt.test_row & opt.test_col:
        row, col = opt.test_row, opt.test_col
    else:
        row_length, col_length = ground_truth.shape[-2:]
        row, col = int(row_length / 2), int(col_length / 2)
    # final_predict = rearrange(final_predict, '(Wk N) T H W -> N (Wk T) H W', N=100)
    # ground_truth = rearrange(ground_truth, '(Wk N) T H W -> N (Wk T) H W', N=100)
    final_predict_rec = final_predict_rec.reshape(B//100*T,100,100)
    ground_truth_rec = ground_truth_rec.reshape(B//100*T,100,100)
    for i in range(50,60):
        for j in range(50,60):
            plt.figure()
            plt.plot(final_predict_rec[:, i, j], 'r-', label='Predicted')
            plt.plot(ground_truth_rec[:, i, j], 'k-', label='GroundTruth')
            plt.legend(loc='upper right')
            plt.savefig('./results/predictions_nogis' + str(i) +'_' + str(j) + '.png')
            plt.close()
    # plt.show()


def train_valid_split(dataloader, val_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(val_size) is float:
        split = int(np.floor(val_size * length))
    elif type(val_size) is int:
        split = val_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    # path = './data/all_data_sliced.h5'
    path = './data/data_git_version.h5'
    gisdir = '/home/chenym/STDenseNet/data/feat_tif'
    traindataset = MilanSlidingWindowDataset(path, gisdir, opt.traffic, opt.test_size, opt.crop,
                                         opt.rows, opt.cols, mode='train', window_size=opt.window_size, input_len=opt.input_len)
    mmn = traindataset.mmn
    testdataset = MilanSlidingWindowDataset(path, gisdir, opt.traffic, opt.test_size, opt.crop,
                                         opt.rows, opt.cols, mode='test', window_size=opt.window_size, input_len=opt.input_len)
    # x_train, y_train, x_test, y_test, mmn = load_data(path, opt.traffic, opt.close_size, opt.period_size,
    #                                                   opt.trend_size,
    #                                                   opt.test_size, opt.nb_flow, opt.height, opt.width, opt.crop,
    #                                                   opt.rows, opt.cols)
    # x_train.append(y_train)
    # x_test.append(y_test)
    # train_data = list(zip(*x_train))
    # test_data = list(zip(*x_test))
    # print(len(train_data), len(test_data))

    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(traindataset, 0.1, shuffle=True)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(traindataset, batch_size=opt.batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(traindataset, batch_size=opt.batch_size, sampler=valid_sampler, pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=opt.batch_size, shuffle=False)

    # get data channels
    channels = [opt.gis_num, opt.input_len]
    model = DenseNet(nb_flows=opt.nb_flow, drop_rate=opt.drop_rate, channels=channels).cuda()
    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.75 * opt.epoch_size),
                                                                      int(0.5 * opt.epoch_size)], gamma=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # print(model)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training...')
    log(opt.model_filename + '.log', '[training]')
    if opt.train:
        train()

    model = torch.load(opt.model_filename + '.optim')
    predict('test')
    plt.figure()
    plt.plot(torch.load(opt.model_filename + '.model').get('train_loss')[1:-1], 'r-')
    plt.legend(labels=['train_loss'], loc='best')
    plt.savefig('./results/train_loss.png')
    plt.figure()
    plt.plot(torch.load(opt.model_filename + '.model').get('valid_loss')[:-1], 'k-')
    plt.legend(labels=['test_loss'], loc='best')
    plt.savefig('./results/test_loss.png')
