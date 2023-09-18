'''
Function:
  Down-Up-CNN training

Main functions:
  * pre_train_Q(): Pre-training of QT net (output qt depth map)
  * pre_train_BD(): Pre-training of MTT net (output mtt depth map and direction map)
  * train_QBD(): Joint training of QT net and MTT net

Note:
  * More training information can be found in the paper.
Author: Aolin Feng
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset, TensorDataset
import itertools

import Model_QBD as model
from Metrics import Load_Pre_VP_Dataset, adjust_learning_rate, validation_QBD, pre_validation, load_pretrain_model

L1_Loss = nn.L1Loss()
L2_Loss = nn.MSELoss()
Cross_Entropy = nn.CrossEntropyLoss()
base_loss = L1_Loss

# Using weight matrix is trying to add the weight of non-zero value of the batch in the loss function,
# which is decided by the ratio of 0 and 1 in the training dataset.
luma_weight_mat = 0.5 * np.array([[1.0, 0.73, 0.15],
                                  [2.43, 0.35, 0.10],
                                  [0.96, 0.23, 0.07],
                                  [0.59, 0.16, 0.05]])
chroma_weight_mat = 0.5 * np.array([[17.83, 0.49, 0.11],
                                    [1.20, 0.25, 0.07],
                                    [0.58, 0.17, 0.05],
                                    [0.38, 0.12, 0.04]])

def loss_func_MSBD(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, isLuma):
    if isLuma:
        weight_mat = luma_weight_mat
    else:
        weight_mat = chroma_weight_mat
    # dire_label_batch_reg's possible values only indlude 0 and 1
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((args.qp-22)/5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((args.qp-22)/5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((args.qp-22)/5)][2]
    if args.qp == 22:
        weight_d0 = 1.0
    return args.lambb0 * base_loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + \
           args.lambb1 * base_loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + \
           args.lambb2 * base_loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :]) + \
           args.lambd0 * base_loss(weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + \
           args.lambd1 * base_loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + \
           args.lambd2 * base_loss(weight_d2 * bd_out_batch2[:, 1:2, :, :], weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + \
           args.lambresb0 * base_loss(weight_d0 * bd_out_batch0[:, 0:1, :, :],
                                      weight_d0 * bt_label_batch[:, 0:1, :, :]) + \
           args.lambresb1 * base_loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]),
                                      weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + \
           args.lambresb2 * base_loss(weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]),
                                      weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))

def loss_func_QBD(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch, bt_label_batch, dire_label_batch_reg, isLuma):
    if isLuma:
        weight_mat = luma_weight_mat
    else:
        weight_mat = chroma_weight_mat
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((args.qp-22)/5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((args.qp-22)/5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((args.qp-22)/5)][2]
    if args.qp == 22:
        weight_d0 = 1.0
    return args.lambq * base_loss(qt_out_batch, qt_label_batch) + \
           args.lambb0 * base_loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + \
           args.lambb1 * base_loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + \
           args.lambb2 * base_loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :]) + \
           args.lambd0 * base_loss(weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + \
           args.lambd1 * base_loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + \
           args.lambd2 * base_loss(weight_d2 * bd_out_batch2[:, 1:2, :, :], weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + \
           args.lambresb0 * base_loss(weight_d0 * bd_out_batch0[:, 0:1, :, :],
                                      weight_d0 * bt_label_batch[:, 0:1, :, :]) + \
           args.lambresb1 * base_loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]),
                                      weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + \
           args.lambresb2 * base_loss(weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]),
                                      weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))

def loss_func_D(dire_out_batch, dire_label_batch):  # b*9*16*16, b*3*16*16 unused
    loss = 0
    dire_out_batch = dire_out_batch.permute((0, 2, 3, 1))
    vec_dire_out_batch = dire_out_batch.reshape((-1, 9))
    for i in range(3):
        vec_dire_out_batch_i = vec_dire_out_batch[:, i*3:(i+1)*3]
        vec_dire_label_batch_i = dire_label_batch[:, i, :, :].reshape(-1)
        loss += Cross_Entropy(vec_dire_out_batch_i, vec_dire_label_batch_i)
    return loss

def paras_print(args):
    print('************************** Parameters *************************')
    print('--jobID', args.jobID)
    print('--inputDir', args.inputDir)
    print('--outDir', args.outDir)
    print('--lr', args.lr)
    print('--dr', args.dr)
    print('--qp', args.qp)
    print('--epoch', args.epoch)
    print('--batchSize', args.batchSize)
    print('--lamb0', args.lamb0)
    print('--lamb1', args.lamb1)
    print('--lamb2', args.lamb2)
    print('************************** Parameters *************************')

def pre_train_Q(args):
    if args.isLuma:
        Net = model.Luma_Q_Net()
        net_Q_path = "/code/Debug/Models/" + "Pre_Luma_Q_QP22.pkl"
        # /ghome/fengal/VVC_Fast_Partition_DP/Models/
        comp = "Luma"
    else:
        Net = model.Chroma_Q_Net()
        net_Q_path = "/code/Debug/Models/" + "Pre_Chroma_Q_QP22.pkl"
        comp = "Chroma"

    # print("Pre Q net path:", net_Q_path)
    # Net = load_pretrain_model(Net, net_Q_path)
    Net = nn.DataParallel(Net).cuda()

    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_loss, epoch_L1_loss, val_L1_loss, val_accu, test_L1_loss, test_accu\n"
        f.write(s)
    # train_loader = Load_Pre_VP_CTU_Dataset(args.inputDir, args.inputDir1, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=0, isLuma=args.isLuma)
    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=0, isLuma=args.isLuma)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=0, isLuma=args.isLuma)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=0, isLuma=args.isLuma)
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("L1 loss", "Accuracy")
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        L1_loss_list = []
        for step, data in enumerate(train_loader):
            input_batch, qt_label_batch = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()
            qt_out_batch = Net(input_batch)

            # loss = weight_loss_QB(qt_out_batch, qt_label_batch, qt_label_batch1, qt_label_batch2)
            loss = L1_Loss(qt_out_batch, qt_label_batch)
            loss.backward()
            optimizer.step()

            L1_loss = L1_Loss(qt_out_batch, qt_label_batch)
            loss_list.append(loss.item())
            L1_loss_list.append(L1_loss.item())
            if step % 1000 == 0:
                print("epoch: %d step: %d [loss] %.6f [L1 loss] %.6f " % (epoch, step, loss.item(), L1_loss.item()))
        epoch_loss = np.mean(loss_list)
        epoch_L1_loss = np.mean(L1_loss_list)
        val_out_info_list = pre_validation(val_loader, Net, 0)  # validation set loss
        test_out_info_list = pre_validation(test_loader, Net, 0)  # test set loss

        print('***********************************************************************'
              '***********************************************************************')
        print("Epoch: %d  Loss: %.6f  L1 Loss %.6f" % (epoch, epoch_loss, epoch_L1_loss))
        print("Val: Loss: %.6f  Acc: %.6f" % (val_out_info_list[0], val_out_info_list[1]))
        print("Test: Loss: %.6f  Acc: %.6f" % (test_out_info_list[0], test_out_info_list[1]))
        print('***********************************************************************'
              '***********************************************************************')
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_loss, epoch_L1_loss, val_out_info_list[0], val_out_info_list[1], test_out_info_list[0], test_out_info_list[1]]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(),
                       os.path.join(out_dir, comp + "_Q_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f.pkl" % (
                       epoch, epoch_loss, val_out_info_list[0], test_out_info_list[0])))

def pre_train_BD(args):
    if args.isLuma:
        Net = model.Luma_MSBD_Net()
        net_B_path = "/code/Debug/Models/" + "Pre_Luma_B_QP22.pkl"
        # /ghome/fengal/VVC_Fast_Partition_DP/Models/
        comp = "Luma"
    else:
        Net = model.Chroma_MSBD_Net()
        net_B_path = "/code/Debug/Models/" + "Pre_Chroma_B_QP22.pkl"
        comp = "Chroma"

    # net_D_path = "/ghome/fengal/VVC_Fast_Partition_DP/Models/Luma" + "_D_QP" + str(args.qp) + ".pkl"
    # print("Pre B net path:", net_B_path)
    # Net = load_pretrain_model(Net, net_B_path)
    Net = nn.DataParallel(Net).cuda()

    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, val_b2_accu, test_b2_accu\n"
        f.write(s)
    # Net_QB.load_state_dict(torch.load('/ghome/fengal/VVC_Fast_Partition_DP/PreModel/luma_qp32_109.pkl'))

    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=2, isLuma=args.isLuma)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=2, isLuma=args.isLuma)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=2, isLuma=args.isLuma)
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("b0 L1 loss", "b1 L1 loss", "b2 L1 loss", "d0 L1 loss", "d1 L1 loss", "d2 L1 loss")
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
        d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
        for step, data in enumerate(train_loader):
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_reg = dire_label_batch_reg.cuda()

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()
            bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net(input_batch, qt_label_batch)

            # loss = weight_loss_func_MSB(bt_out_batch0, bt_out_batch1, bt_out_batch2, bt_label_batch, bt_label_batch1, bt_label_batch2)
            loss = loss_func_MSBD(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, args.isLuma)
            loss.backward()
            optimizer.step()

            b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
            b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
            b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
            d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
            d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
            d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

            loss_list.append(loss.item())
            b0_L1_loss_list.append(b0_L1_loss.item())
            b1_L1_loss_list.append(b1_L1_loss.item())
            b2_L1_loss_list.append(b2_L1_loss.item())
            d0_L1_loss_list.append(d0_L1_loss.item())
            d1_L1_loss_list.append(d1_L1_loss.item())
            d2_L1_loss_list.append(d2_L1_loss.item())
            if step % 1000 == 0:
                print("epoch: %d step: %d [loss] %.6f [b] %.6f %.6f %.6f [d] %.6f %.6f %.6f" %(epoch, step, loss.item(),
                      b0_L1_loss.item(), b1_L1_loss.item(), b2_L1_loss.item(), d0_L1_loss.item(), d1_L1_loss.item(), d2_L1_loss.item()))
        epoch_loss = np.mean(loss_list)
        epoch_b0_L1_loss = np.mean(b0_L1_loss_list)
        epoch_b1_L1_loss = np.mean(b1_L1_loss_list)
        epoch_b2_L1_loss = np.mean(b2_L1_loss_list)
        epoch_d0_L1_loss = np.mean(d0_L1_loss_list)
        epoch_d1_L1_loss = np.mean(d1_L1_loss_list)
        epoch_d2_L1_loss = np.mean(d2_L1_loss_list)
        val_out_info_list = pre_validation(val_loader, Net, 1, args.qp)  # validation set loss
        test_out_info_list = pre_validation(test_loader, Net, 1, args.qp)  # test set loss

        print('*****************************************************************'
              '*****************************************************************')
        # print("Epoch:", epoch, " Loss:", epoch_loss)
        print("Epoch: %d  Loss: %.6f %.6f %.6f" %(epoch, epoch_loss, val_out_info_list[12], test_out_info_list[12]))
        print("Train L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss,
                                                                  epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss))
        # print("Val Loss:", val_out_info_list[12])
        print("Val L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(val_out_info_list[0], val_out_info_list[1], val_out_info_list[2],
                                                                val_out_info_list[3], val_out_info_list[4], val_out_info_list[5]))
        print("Val Accu: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(val_out_info_list[6], val_out_info_list[7], val_out_info_list[8],
                                                                  val_out_info_list[9], val_out_info_list[10], val_out_info_list[11]))
        # print("Test Loss:", test_out_info_list[12])
        print("Test L1: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(test_out_info_list[0], test_out_info_list[1], test_out_info_list[2],
                                                                 test_out_info_list[3], test_out_info_list[4], test_out_info_list[5]))
        print("Test Accu: [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(test_out_info_list[6], test_out_info_list[7], test_out_info_list[8],
                                                                   test_out_info_list[9], test_out_info_list[10], test_out_info_list[11]))


        print('******************************************************************'
              '******************************************************************')
        # save training loss
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(),
                       os.path.join(out_dir, comp + "_BD_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f.pkl" % (
                       epoch, epoch_loss, val_out_info_list[12], test_out_info_list[12])))

def train_QBD(args):
    if args.isLuma:
        Net_Q = model.Luma_Q_Net()
        Net_BD = model.Luma_MSBD_Net()
        net_Q_path = "/code/Debug/Models/" + "Luma_Q_" + str(args.qp) + ".pkl"
        net_BD_path = "/code/Debug/Models/" + "Luma_BD_" + str(args.qp) + ".pkl"
        # /ghome/fengal/VVC_Fast_Partition_DP/Models/
        comp = "Luma"
    else:
        Net_Q = model.Chroma_Q_Net()
        Net_BD = model.Chroma_MSBD_Net()
        net_Q_path = "/code/Debug/Models/" + "Chroma_Q_" + str(args.qp) + ".pkl"
        net_BD_path = "/code/Debug/Models/" + "Chroma_BD_" + str(args.qp) + ".pkl"
        comp = "Chroma"

    # Net = nn.DataParallel(Net).cuda()
    print("Pre Q net path:", net_Q_path)
    print("Pre BD net path:", net_BD_path)
    Net_Q = load_pretrain_model(Net_Q, net_Q_path)
    Net_BD = load_pretrain_model(Net_BD, net_BD_path)
    Net_Q = nn.DataParallel(Net_Q).cuda()
    Net_BD = nn.DataParallel(Net_BD).cuda()

    out_dir = os.path.join(args.outDir, args.jobID)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'loss.txt')
    with open(log_dir, 'a') as f:
        s = "epoch_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, val_b2_accu, test_b2_accu\n"
        f.write(s)

    train_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=0, PredID=2, isLuma=args.isLuma)
    val_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=1, PredID=2, isLuma=args.isLuma)
    test_loader = Load_Pre_VP_Dataset(args.inputDir, QP=args.qp, batchSize=args.batchSize, datasetID=2, PredID=2, isLuma=args.isLuma)
    optimizer = torch.optim.Adam(itertools.chain(Net_Q.parameters(), Net_BD.parameters()), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('Start Training ...')
    print("q L1 loss", "b0 L1 loss", "b1 L1 loss", "b2 L1 loss", "d0 L1 loss", "d1 L1 loss", "d2 L1 loss")
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch, args.dr)
        loss_list = []
        q_L1_loss_list = []
        b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
        d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
        for step, data in enumerate(train_loader):
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_reg = dire_label_batch_reg.cuda()

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()
            qt_out_batch = Net_Q(input_batch)
            bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net_BD(input_batch, qt_out_batch)

            # loss = weight_loss_func_MSB(bt_out_batch0, bt_out_batch1, bt_out_batch2, bt_label_batch, bt_label_batch1, bt_label_batch2)
            loss = loss_func_QBD(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch, bt_label_batch, dire_label_batch_reg, args.isLuma)
            loss.backward()
            optimizer.step()

            q_L1_loss = L1_Loss(qt_out_batch, qt_label_batch)
            b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
            b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
            b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
            d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
            d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
            d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

            loss_list.append(loss.item())
            q_L1_loss_list.append(q_L1_loss.item())
            b0_L1_loss_list.append(b0_L1_loss.item())
            b1_L1_loss_list.append(b1_L1_loss.item())
            b2_L1_loss_list.append(b2_L1_loss.item())
            d0_L1_loss_list.append(d0_L1_loss.item())
            d1_L1_loss_list.append(d1_L1_loss.item())
            d2_L1_loss_list.append(d2_L1_loss.item())
            if step % 1000 == 0:
                print("epoch: %d step: %d [loss] %.6f [q] %.6f [b] %.6f %.6f %.6f [d] %.6f %.6f %.6f" %(epoch, step, loss.item(), q_L1_loss.item(),
                      b0_L1_loss.item(), b1_L1_loss.item(), b2_L1_loss.item(), d0_L1_loss.item(), d1_L1_loss.item(), d2_L1_loss.item()))
        epoch_loss = np.mean(loss_list)
        epoch_q_L1_loss = np.mean(q_L1_loss_list)
        epoch_b0_L1_loss = np.mean(b0_L1_loss_list)
        epoch_b1_L1_loss = np.mean(b1_L1_loss_list)
        epoch_b2_L1_loss = np.mean(b2_L1_loss_list)
        epoch_d0_L1_loss = np.mean(d0_L1_loss_list)
        epoch_d1_L1_loss = np.mean(d1_L1_loss_list)
        epoch_d2_L1_loss = np.mean(d2_L1_loss_list)
        val_out_info_list = validation_QBD(val_loader, Net_Q, Net_BD, args.qp)  # validation set loss
        test_out_info_list = validation_QBD(test_loader, Net_Q, Net_BD, args.qp)  # test set loss

        print('*****************************************************************'
              '*****************************************************************')
        # print("Epoch:", epoch, " Loss:", epoch_loss)
        print("Epoch: %d  Loss: %.6f %.6f %.6f" %(epoch, epoch_loss, val_out_info_list[14], test_out_info_list[14]))
        print("Train L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(epoch_q_L1_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss,
                                                                           epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss))

        print("Val L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(val_out_info_list[0], val_out_info_list[1], val_out_info_list[2],
                                                                val_out_info_list[3], val_out_info_list[4], val_out_info_list[5], val_out_info_list[6]))
        print("Val Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(val_out_info_list[7], val_out_info_list[8], val_out_info_list[9],
                                                                  val_out_info_list[10], val_out_info_list[11], val_out_info_list[12], val_out_info_list[13]))

        print("Test L1: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(test_out_info_list[0], test_out_info_list[1], test_out_info_list[2],
                                                                 test_out_info_list[3], test_out_info_list[4], test_out_info_list[5], test_out_info_list[6]))
        print("Test Accu: [Q] %.6f [B] %.6f %.6f %.6f [D] %.6f %.6f %.6f" %(test_out_info_list[7], test_out_info_list[8], test_out_info_list[9],
                                                                test_out_info_list[10], test_out_info_list[11], test_out_info_list[12], test_out_info_list[13]))


        print('******************************************************************'
              '******************************************************************')
        # save training loss
        with open(log_dir, 'a') as f:
            for s in [epoch_loss, epoch_q_L1_loss, epoch_b0_L1_loss, epoch_b1_L1_loss, epoch_b2_L1_loss, epoch_d0_L1_loss, epoch_d1_L1_loss, epoch_d2_L1_loss]:
                f.write(str(s))
                f.write(',')
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(Net_Q.state_dict(),
                       os.path.join(out_dir, comp + "_Q_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f_%.4f.pkl" % (
                       epoch, epoch_q_L1_loss, epoch_loss, val_out_info_list[12], test_out_info_list[12])))
            torch.save(Net_BD.state_dict(),
                       os.path.join(out_dir, comp + "_BD_" + str(args.qp) + "_epoch_%d_%.4f_%.4f_%.4f_%.4f.pkl" % (
                       epoch, epoch_b2_L1_loss, epoch_loss, val_out_info_list[12], test_out_info_list[12])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument("--isLuma", dest='isLuma', action="store_true")
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--inputDir1', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default='/output/')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dr', default=20, type=int, help='decay rate of lr')
    parser.add_argument('--qp', default=22, type=int, help='quantization step')
    parser.add_argument('--epoch', default=100, type=int, help='number of total epoch')
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--lamb0', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lamb1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lamb2', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--predID', default=0, type=int, help='[0 1 2] [qt bt direction]')

    parser.add_argument('--lambq', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambb0', default=0.8, type=float, help='weight of loss function')
    parser.add_argument('--lambb1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambb2', default=1.2, type=float, help='weight of loss function')
    parser.add_argument('--lambd0', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambd1', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambd2', default=1.0, type=float, help='weight of loss function')
    parser.add_argument('--lambresb0', default=0.5, type=float, help='weight of loss function')
    parser.add_argument('--lambresb1', default=0.5, type=float, help='weight of loss function')
    parser.add_argument('--lambresb2', default=0.5, type=float, help='weight of loss function')
    args = parser.parse_args()

    paras_print(args)
    if args.predID == 0:
        pre_train_Q(args)
    elif args.predID == 1:
        pre_train_BD(args)
    elif args.predID == 2:
        train_QBD(args)
    else:
        print("Unknown predID!!!")
    # train_BD(args)
    '''
    reference 
    python /code/Train_QBD.py --inputDir /data/FengAolin/VTM10_Partition --outDir /output/ --lr 5e-4 --dr 30 --epoch 300 --qp 22 2>&1 | tee /output/test.txt
	startdocker -c "python3 -u /ghome/fengal/VVC_Fast_Partition_DP/Train_QBD.py --predID 0 --addNoise 0 --qp22 --lr 5e-4 --dr 30 --epoch 300 --batchSize 400 --jobID $PBS_JOBID --inputDir /gdata/fengal/VTM10_Partition --outDir /ghome/fengal/VVC_Fast_Partition_DP/Output" -P /ghome/fengal -D /gdata/fengal bit:5000/zj_py3.6_torch1.3
    python /code/Debug/Train_QBD.py --predID 0 --qp 32 --lr 2e-4 --dr 30 --epoch 200 --batchSize 256 --jobID LumaD22 --inputDir /data/FengAolin/VTM10_Partition_LMCS0_LFNST0 --outDir /output/
    '''




