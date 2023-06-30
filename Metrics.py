'''
Function：
  Related metrics used in other file

Note:
  * Not all the functions in this files are used. Check this file when other files refer it.

Author: Aolin Feng
'''

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from Map2Partition import get_sequence_partition_for_VTM

L1_Loss = nn.L1Loss()
Cross_Entropy = nn.CrossEntropyLoss()


def Mul_Scale_L1Loss(pred_map, label_map):
    pred_map_max1 = F.max_pool2d(pred_map, 8)
    pred_map_max2 = F.max_pool2d(pred_map, 4)
    pred_map_max4 = F.max_pool2d(pred_map, 2)

    pred_map_min1 = -F.max_pool2d(-pred_map, 8)
    pred_map_min2 = -F.max_pool2d(-pred_map, 4)
    pred_map_min4 = -F.max_pool2d(-pred_map, 2)

    label_map_max1 = F.max_pool2d(label_map, 8)
    label_map_max2 = F.max_pool2d(label_map, 4)
    label_map_max4 = F.max_pool2d(label_map, 2)

    label_map_min1 = -F.max_pool2d(-label_map, 8)
    label_map_min2 = -F.max_pool2d(-label_map, 4)
    label_map_min4 = -F.max_pool2d(-label_map, 2)

    # MS_L1_Loss = L1_Loss(pred_map_max1, label_map_max1) * 1/170.0 + L1_Loss(pred_map_max2, label_map_max2) * 4/170.0+ L1_Loss(pred_map_max4, label_map_max4) * 16/170.0 + \
    #             L1_Loss(pred_map_min1, label_map_min1) * 1/170.0 + L1_Loss(pred_map_min2, label_map_min2) * 4/170.0 + L1_Loss(pred_map_min4, label_map_min4) * 16/170.0 + \
    #             2.0 * L1_Loss(pred_map, label_map) * 64/170.0
    MS_L1_Loss = L1_Loss(pred_map_max1, label_map_max1) * 1/30.0 + L1_Loss(pred_map_max2, label_map_max2) * 2/30.0 + L1_Loss(pred_map_max4, label_map_max4) * 4/30.0 + \
                L1_Loss(pred_map_min1, label_map_min1) * 1/30.0 + L1_Loss(pred_map_min2, label_map_min2) * 2/30.0 + L1_Loss(pred_map_min4, label_map_min4) * 4/30.0 + \
                2.0 * L1_Loss(pred_map, label_map) * 8/30.0

    return MS_L1_Loss


def loss_func_D(dire_out_batch, dire_label_batch):  # b*9*16*16, b*3*16*16
    loss = 0
    dire_out_batch = dire_out_batch.permute((0, 2, 3, 1))
    vec_dire_out_batch = dire_out_batch.reshape((-1, 9))
    for i in range(3):
        vec_dire_out_batch_i = vec_dire_out_batch[:, i*3:(i+1)*3]
        vec_dire_label_batch_i = dire_label_batch[:, i, :, :].reshape(-1)
        loss += Cross_Entropy(vec_dire_out_batch_i, vec_dire_label_batch_i)
    return loss

def adjust_learning_rate(lr, optimizer, epoch, decay_rate):
    adj_lr = lr * (0.5 ** (epoch // decay_rate))
    if adj_lr > 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = adj_lr


#****************************************************************************************************************
# Pre Train
#****************************************************************************************************************

def Load_Pre_Multi_VP_Dataset(path, QP, batchSize, datasetID=0, PredID=0 ,isLuma=True):
    # datasetID [train validation test]; PredID [QT BT Direction]
    # add variance map to the input
    if isLuma:
        comp = 'Luma'
    else:
        comp = 'Chroma'
    tr_val_test = ['Train', 'Validate', 'TestSub']
    dataset_type = tr_val_test[datasetID]

    print('Start loading pre-train ' + comp + ' ' + dataset_type + ' dataset...')
    if isLuma:  # luma input
        input_path = os.path.join(path, dataset_type + '_Y_Block68.npy')
        input_path1 = os.path.join(path, dataset_type + '_Y_Variance.npy')
        print('input path0:', input_path)
        # print('input path1:', input_path1)
        input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))
        # input_batch1 = F.interpolate(torch.FloatTensor(np.expand_dims(np.load(input_path1), 1)), scale_factor=4)
        # input_batch = torch.cat([input_batch, input_batch1], 1)
        # del input_batch1

    else:  # chroma input
        input_path = os.path.join(path, dataset_type + '_U_Block34.npy')
        input_path1 = os.path.join(path, dataset_type + '_V_Block34.npy')
        input_path2 = os.path.join(path, dataset_type + '_U_Variance.npy')
        input_path3 = os.path.join(path, dataset_type + '_V_Variance.npy')
        print('input path0:', input_path)
        print('input path1:', input_path1)
        input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))
        input_batch1 = torch.FloatTensor(np.expand_dims(np.load(input_path1), 1))
        input_batch = torch.cat([input_batch, input_batch1], 1)
        del input_batch1

    print('input_batch.shape:', input_batch.shape)
    if PredID == 0:  # Q
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        qt_label_path1 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth1_Block8.npy')
        qt_label_path2 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth2_Block8.npy')
        print('qt_label path:', qt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        qt_label_batch1 = torch.FloatTensor(np.expand_dims(np.load(qt_label_path1), 1) - 1)  # qt depth start form 1
        qt_label_batch2 = torch.FloatTensor(np.expand_dims(np.load(qt_label_path2), 1) - 1)  # qt depth start form 1
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print("Creating Multi Q data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        dataset = TensorDataset(input_batch, qt_label_batch, qt_label_batch1, qt_label_batch2)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 1:  # B
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        # qt_label_path1 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth1_Block8.npy')
        # qt_label_path2 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth2_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        bt_label_path1 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth1_Block16.npy')
        bt_label_path2 = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth2_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        # qt_label_batch1 = torch.FloatTensor(np.expand_dims(np.load(qt_label_path1), 1) - 1)  # qt depth start form 1
        # qt_label_batch2 = torch.FloatTensor(np.expand_dims(np.load(qt_label_path2), 1) - 1)  # qt depth start form 1
        bt_label_batch = torch.FloatTensor(np.expand_dims(np.load(bt_label_path), 1))
        bt_label_batch1 = torch.FloatTensor(np.expand_dims(np.load(bt_label_path1), 1))
        bt_label_batch2 = torch.FloatTensor(np.expand_dims(np.load(bt_label_path2), 1))
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        # norm_input_batch = block_qtnode_norm(qt_map=qt_label_batch, block=input_batch, isLuma=isLuma)
        print("Creating Multi B data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        # bt_label_batch = bt_label_batch[0:1157480]
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch, bt_label_batch1, bt_label_batch2)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 2:  # D
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        dire_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSdirection_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        print('direction_label path:', dire_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        bt_label_batch = torch.FloatTensor(np.load(bt_label_path))
        dire_label_batch_reg = torch.LongTensor(np.load(dire_label_path))
        dire_label_batch_cla = torch.LongTensor(
            torch.where(dire_label_batch_reg == -1, torch.full_like(dire_label_batch_reg, 2), dire_label_batch_reg))
        del dire_label_batch_reg
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        print('dire_label_batch.shape:', dire_label_batch_cla.shape)

        print("Creating D data loader...")
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch, dire_label_batch_cla)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)
    else:
        print("Unknown Dataset!!!")
        return

    return dataLoader

def Load_Pre_VP_CTU_Dataset(ori_path, ctu_path, QP, batchSize, datasetID=0, PredID=0 ,isLuma=True):
    # ctu_path save CTU partition label
    # datasetID [train validation test]; PredID [QT BT Direction]
    # add variance map to the input
    if isLuma:
        comp = 'Luma'
    else:
        comp = 'Chroma'
    tr_val_test = ['Train', 'Validate', 'TestSub']
    dataset_type = tr_val_test[datasetID]

    print('Start loading pre-train ' + comp + ' ' + dataset_type + ' dataset...')
    # if isLuma:  # luma input
    input_path = os.path.join(ori_path, dataset_type + '_Y_Block68.npy')
    print('input path0:', input_path)
    input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))

    if not isLuma:  # chroma input
        input_path1 = os.path.join(ori_path, dataset_type + '_U_Block34.npy')
        input_path2 = os.path.join(ori_path, dataset_type + '_V_Block34.npy')
        print('input path1:', input_path1)
        print('input path2:', input_path2)
        input_batch = F.max_pool2d(input_batch, 2)
        input_batch1 = torch.FloatTensor(np.expand_dims(np.load(input_path1), 1))
        input_batch2 = torch.FloatTensor(np.expand_dims(np.load(input_path2), 1))
        input_batch = torch.cat([input_batch, input_batch1, input_batch2], 1)
        del input_batch1, input_batch2

    if datasetID == 0:  # use CTU partition label for training
        path = ctu_path
    else:
        path = ori_path

    if PredID == 0:  # Q
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        print('qt_label path:', qt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print("Creating Q data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        input_batch = input_batch[0:qt_label_batch.shape[0]]
        dataset = TensorDataset(input_batch, qt_label_batch)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 1:  # QB
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        bt_label_batch = torch.FloatTensor(np.expand_dims(np.load(bt_label_path), 1))
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        # norm_input_batch = block_qtnode_norm(qt_map=qt_label_batch, block=input_batch, isLuma=isLuma)
        print("Creating BD data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        # bt_label_batch = bt_label_batch[0:1157480]
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 2:  # QBD for MSBD training
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        dire_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSdirection_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        print('direction_label path:', dire_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        # bt_label_batch = torch.FloatTensor(np.load(bt_label_path))
        # dire_label_batch_reg = torch.FloatTensor(np.load(dire_label_path))
        # dire_label_batch_cla = torch.LongTensor(
        #    torch.where(dire_label_batch_reg == -1, torch.full_like(dire_label_batch_reg, 2), dire_label_batch_reg))
        # del dire_label_batch_reg

        bt_label_batch = torch.FloatTensor(np.load(bt_label_path))
        dire_label_batch_reg = torch.FloatTensor(np.load(dire_label_path))
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        print('dire_label_batch.shape:', dire_label_batch_reg.shape)
        print("Creating QBD data loader...")
        input_batch = input_batch[0:qt_label_batch.shape[0]]
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)
    else:
        print("Unknown Dataset!!!")
        return

    return dataLoader

def Load_Pre_VP_Dataset(path, QP, batchSize, datasetID=0, PredID=0 ,isLuma=True):
    # datasetID [train validation test]; PredID [QT BT Direction]
    # add variance map to the input
    if isLuma:
        comp = 'Luma'
    else:
        comp = 'Chroma'
    tr_val_test = ['Train', 'Validate', 'TestSub']
    dataset_type = tr_val_test[datasetID]

    print('Start loading pre-train ' + comp + ' ' + dataset_type + ' dataset...')
    # if isLuma:  # luma input
    input_path = os.path.join(path, dataset_type + '_Y_Block68.npy')
    print('input path0:', input_path)
    input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))

    if not isLuma:  # chroma input
        input_path1 = os.path.join(path, dataset_type + '_U_Block34.npy')
        input_path2 = os.path.join(path, dataset_type + '_V_Block34.npy')
        print('input path1:', input_path1)
        print('input path2:', input_path2)
        input_batch = F.max_pool2d(input_batch, 2)
        input_batch1 = torch.FloatTensor(np.expand_dims(np.load(input_path1), 1))
        input_batch2 = torch.FloatTensor(np.expand_dims(np.load(input_path2), 1))
        input_batch = torch.cat([input_batch, input_batch1, input_batch2], 1)
        del input_batch1, input_batch2

    print('input_batch.shape:', input_batch.shape)
    if PredID == 0:  # Q
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        print('qt_label path:', qt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print("Creating Q data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        dataset = TensorDataset(input_batch, qt_label_batch)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 1:  # QB
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        bt_label_batch = torch.FloatTensor(np.expand_dims(np.load(bt_label_path), 1))
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        # norm_input_batch = block_qtnode_norm(qt_map=qt_label_batch, block=input_batch, isLuma=isLuma)
        print("Creating BD data loader...")
        # input_batch = input_batch[0:1157480]
        # qt_label_batch = qt_label_batch[0:1157480]
        # bt_label_batch = bt_label_batch[0:1157480]
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    elif PredID == 2:  # QBD for MSBD training
        qt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSBTdepth_Block16.npy')
        dire_label_path = os.path.join(path, dataset_type + '_' + comp + '_QP' + str(QP) + '_MSdirection_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        print('direction_label path:', dire_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        # bt_label_batch = torch.FloatTensor(np.load(bt_label_path))
        # dire_label_batch_reg = torch.FloatTensor(np.load(dire_label_path))
        # dire_label_batch_cla = torch.LongTensor(
        #    torch.where(dire_label_batch_reg == -1, torch.full_like(dire_label_batch_reg, 2), dire_label_batch_reg))
        # del dire_label_batch_reg

        bt_label_batch = torch.FloatTensor(np.load(bt_label_path))
        dire_label_batch_reg = torch.FloatTensor(np.load(dire_label_path))
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)
        print('dire_label_batch.shape:', dire_label_batch_reg.shape)
        print("Creating QBD data loader...")
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)
    else:
        print("Unknown Dataset!!!")
        return

    return dataLoader

weight_mat = 0.5 * np.array([[1.0, 0.73, 0.15],
                             [2.43, 0.35, 0.10],
                             [0.96, 0.23, 0.07],
                             [0.59, 0.16, 0.05]])
# weight_mat = 0.5 * np.array([[17.83, 0.49, 0.11],
#                              [1.20, 0.25, 0.07],
#                              [0.58, 0.17, 0.05],
#                              [0.38, 0.12, 0.04]])

def loss_func_MSBD_val(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, qp):
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((qp-22)/5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((qp-22)/5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((qp-22)/5)][2]
    if qp == 22:
        weight_d0 = 1.0
    return 0.8 * L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + \
           1.0 * L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + \
           1.2 * L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :]) + \
           1.0 * L1_Loss(weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + \
           1.0 * L1_Loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + \
           1.0 * L1_Loss(weight_d2 * bd_out_batch2[:, 1:2, :, :], weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + \
           0.5 * L1_Loss(weight_d0 * bd_out_batch0[:, 0:1, :, :],
                                      weight_d0 * bt_label_batch[:, 0:1, :, :]) + \
           0.5 * L1_Loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]),
                                      weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + \
           0.5 * L1_Loss(weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]),
                                      weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))

def loss_func_QBD_val(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch, bt_label_batch, dire_label_batch_reg, qp):
    weight_d0 = dire_label_batch_reg[:, 0:1, :, :] * dire_label_batch_reg[:, 0:1, :, :] + weight_mat[int((qp-22)/5)][0]
    weight_d1 = dire_label_batch_reg[:, 1:2, :, :] * dire_label_batch_reg[:, 1:2, :, :] + weight_mat[int((qp-22)/5)][1]
    weight_d2 = dire_label_batch_reg[:, 2:3, :, :] * dire_label_batch_reg[:, 2:3, :, :] + weight_mat[int((qp-22)/5)][2]
    if qp == 22:
        weight_d0 = 1.0
    return 1.0 * L1_Loss(qt_out_batch, qt_label_batch) + \
           0.8 * L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :]) + \
           1.0 * L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :]) + \
           1.2 * L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :]) + \
           1.0 * L1_Loss(weight_d0 * bd_out_batch0[:, 1:2, :, :], weight_d0 * dire_label_batch_reg[:, 0:1, :, :]) + \
           1.0 * L1_Loss(weight_d1 * bd_out_batch1[:, 1:2, :, :], weight_d1 * dire_label_batch_reg[:, 1:2, :, :]) + \
           1.0 * L1_Loss(weight_d2 * bd_out_batch2[:, 1:2, :, :], weight_d2 * dire_label_batch_reg[:, 2:3, :, :]) + \
           0.5 * L1_Loss(weight_d0 * bd_out_batch0[:, 0:1, :, :],
                                      weight_d0 * bt_label_batch[:, 0:1, :, :]) + \
           0.5 * L1_Loss(weight_d1 * (bd_out_batch1[:, 0:1, :, :] - bd_out_batch0[:, 0:1, :, :]),
                                      weight_d1 * (bt_label_batch[:, 1:2, :, :] - bt_label_batch[:, 0:1, :, :])) + \
           0.5 * L1_Loss(weight_d2 * (bd_out_batch2[:, 0:1, :, :] - bd_out_batch1[:, 0:1, :, :]),
                                      weight_d2 * (bt_label_batch[:, 2:3, :, :] - bt_label_batch[:, 1:2, :, :]))

@torch.no_grad()
def pre_validation(val_loader, Net, predID, qp=22):
    if predID == 0:  # Q
        with torch.no_grad():
            L1_loss_list = []
            accu_list = []
            for step, data in enumerate(val_loader):
                input_batch, qt_label_batch = data
                input_batch = input_batch.cuda()
                qt_label_batch = qt_label_batch.cuda()
                qt_out_batch = Net(input_batch)
                qt_accuracy = torch.sum(torch.round(qt_out_batch) == qt_label_batch).item() / float(qt_out_batch.numel())
                L1_loss = L1_Loss(qt_out_batch, qt_label_batch)
                L1_loss_list.append(L1_loss.item())
                accu_list.append(qt_accuracy)
                del input_batch, qt_label_batch, qt_out_batch
        return [np.mean(L1_loss_list), np.mean(accu_list)]
    elif predID == 1:  # BD
        with torch.no_grad():
            val_loss_list = []
            b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
            d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
            b0_accu_list, b1_accu_list, b2_accu_list = [], [], []
            d0_accu_list, d1_accu_list, d2_accu_list = [], [], []
            for step, data in enumerate(val_loader):
                input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
                input_batch = input_batch.cuda()
                qt_label_batch = qt_label_batch.cuda()
                bt_label_batch = bt_label_batch.cuda()
                dire_label_batch_reg = dire_label_batch_reg.cuda()
                bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net(input_batch, qt_label_batch)
                val_loss = loss_func_MSBD_val(bd_out_batch0, bd_out_batch1, bd_out_batch2, bt_label_batch, dire_label_batch_reg, qp)
                val_loss_list.append(val_loss.item())

                b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
                b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
                b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
                d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
                d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
                d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

                b0_accuracy = torch.sum(
                    torch.round(bd_out_batch0[:, 0:1, :, :]) == bt_label_batch[:, 0:1, :, :]).item() / float(
                    bd_out_batch0[:, 0:1, :, :].numel())
                b1_accuracy = torch.sum(
                    torch.round(bd_out_batch1[:, 0:1, :, :]) == bt_label_batch[:, 1:2, :, :]).item() / float(
                    bd_out_batch1[:, 0:1, :, :].numel())
                b2_accuracy = torch.sum(
                    torch.round(bd_out_batch2[:, 0:1, :, :]) == bt_label_batch[:, 2:3, :, :]).item() / float(
                    bd_out_batch2[:, 0:1, :, :].numel())
                d0_accuracy = torch.sum(
                    torch.round(bd_out_batch0[:, 1:2, :, :]) == dire_label_batch_reg[:, 0:1, :, :]).item() / float(
                    bd_out_batch0[:, 1:2, :, :].numel())
                d1_accuracy = torch.sum(
                    torch.round(bd_out_batch1[:, 1:2, :, :]) == dire_label_batch_reg[:, 1:2, :, :]).item() / float(
                    bd_out_batch1[:, 1:2, :, :].numel())
                d2_accuracy = torch.sum(
                    torch.round(bd_out_batch2[:, 1:2, :, :]) == dire_label_batch_reg[:, 2:3, :, :]).item() / float(
                    bd_out_batch2[:, 1:2, :, :].numel())

                b0_L1_loss_list.append(b0_L1_loss.item())
                b1_L1_loss_list.append(b1_L1_loss.item())
                b2_L1_loss_list.append(b2_L1_loss.item())
                d0_L1_loss_list.append(d0_L1_loss.item())
                d1_L1_loss_list.append(d1_L1_loss.item())
                d2_L1_loss_list.append(d2_L1_loss.item())

                b0_accu_list.append(b0_accuracy)
                b1_accu_list.append(b1_accuracy)
                b2_accu_list.append(b2_accuracy)
                d0_accu_list.append(d0_accuracy)
                d1_accu_list.append(d1_accuracy)
                d2_accu_list.append(d2_accuracy)

                del input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg, bd_out_batch0, bd_out_batch1, bd_out_batch2
        return [np.mean(b0_L1_loss_list), np.mean(b1_L1_loss_list), np.mean(b2_L1_loss_list),
                np.mean(d0_L1_loss_list), np.mean(d1_L1_loss_list), np.mean(d2_L1_loss_list),
                np.mean(b0_accu_list), np.mean(b1_accu_list), np.mean(b2_accu_list),
                np.mean(d0_accu_list), np.mean(d1_accu_list), np.mean(d2_accu_list), np.mean(val_loss_list)]

    elif predID == 2:  # D
        with torch.no_grad():
            loss_list = []
            L1_loss_list = []
            d0_accu_list, d1_accu_list, d2_accu_list = [], [], []
            for step, data in enumerate(val_loader):
                input_batch, qt_label_batch, bt_label_batch, dire_label_batch_cla = data
                input_batch = input_batch.cuda()
                qt_label_batch = qt_label_batch.cuda()
                bt_label_batch = bt_label_batch.cuda()
                dire_label_batch_cla = dire_label_batch_cla.cuda()

                dire_out_batch = Net(input_batch, qt_label_batch, bt_label_batch)
                loss = loss_func_D(dire_out_batch, dire_label_batch_cla)
                num = dire_out_batch.shape[0]
                dire_out_batch_cla = torch.zeros(num, 3, 16, 16).cuda()
                for i in range(3):
                    dire_out_batch_cla[:, i, :, :] = torch.argmax(dire_out_batch[:, i * 3:(i + 1) * 3, :, :], dim=1)
                d0_accu = torch.sum(dire_out_batch_cla[:, 0, :, :] == dire_label_batch_cla[:, 0, :, :]).item() \
                                 / float(dire_out_batch_cla[:, 0, :, :].numel())
                d1_accu = torch.sum(dire_out_batch_cla[:, 1, :, :] == dire_label_batch_cla[:, 1, :, :]).item() \
                                 / float(dire_out_batch_cla[:, 1, :, :].numel())
                d2_accu = torch.sum(dire_out_batch_cla[:, 2, :, :] == dire_label_batch_cla[:, 2, :, :]).item() \
                                 / float(dire_out_batch_cla[:, 2, :, :].numel())
                dire_L1_loss = L1_Loss(dire_out_batch_cla.float(), dire_label_batch_cla.float())
                loss_list.append(loss.item())
                L1_loss_list.append(dire_L1_loss.item())
                d0_accu_list.append(d0_accu)
                d1_accu_list.append(d1_accu)
                d2_accu_list.append(d2_accu)

                del input_batch, qt_label_batch, bt_label_batch, dire_label_batch_cla, dire_out_batch, dire_out_batch_cla
        return np.mean(loss_list), np.mean(L1_loss_list), np.mean(d0_accu_list), np.mean(d1_accu_list), np.mean(d2_accu_list)
    else:
        print("Unknown Validation !!!")
        return

@torch.no_grad()
def validation_QBD(val_loader, Net_Q, Net_BD, qp=22):
    with torch.no_grad():
        val_loss_list = []
        q_L1_loss_list, q_accu_list = [], []
        b0_L1_loss_list, b1_L1_loss_list, b2_L1_loss_list = [], [], []
        d0_L1_loss_list, d1_L1_loss_list, d2_L1_loss_list = [], [], []
        b0_accu_list, b1_accu_list, b2_accu_list = [], [], []
        d0_accu_list, d1_accu_list, d2_accu_list = [], [], []
        for step, data in enumerate(val_loader):
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_reg = dire_label_batch_reg.cuda()

            qt_out_batch = Net_Q(input_batch)
            bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net_BD(input_batch, qt_out_batch)

            val_loss = loss_func_QBD_val(qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2, qt_label_batch,
                                         bt_label_batch, dire_label_batch_reg, qp)
            val_loss_list.append(val_loss.item())

            q_L1_loss = L1_Loss(qt_out_batch, qt_label_batch)
            b0_L1_loss = L1_Loss(bd_out_batch0[:, 0:1, :, :], bt_label_batch[:, 0:1, :, :])
            b1_L1_loss = L1_Loss(bd_out_batch1[:, 0:1, :, :], bt_label_batch[:, 1:2, :, :])
            b2_L1_loss = L1_Loss(bd_out_batch2[:, 0:1, :, :], bt_label_batch[:, 2:3, :, :])
            d0_L1_loss = L1_Loss(bd_out_batch0[:, 1:2, :, :], dire_label_batch_reg[:, 0:1, :, :])
            d1_L1_loss = L1_Loss(bd_out_batch1[:, 1:2, :, :], dire_label_batch_reg[:, 1:2, :, :])
            d2_L1_loss = L1_Loss(bd_out_batch2[:, 1:2, :, :], dire_label_batch_reg[:, 2:3, :, :])

            q_accuracy = torch.sum(torch.round(qt_out_batch) == qt_label_batch).item() / float(qt_out_batch.numel())
            b0_accuracy = torch.sum(
                torch.round(bd_out_batch0[:, 0:1, :, :]) == bt_label_batch[:, 0:1, :, :]).item() / float(
                bd_out_batch0[:, 0:1, :, :].numel())
            b1_accuracy = torch.sum(
                torch.round(bd_out_batch1[:, 0:1, :, :]) == bt_label_batch[:, 1:2, :, :]).item() / float(
                bd_out_batch1[:, 0:1, :, :].numel())
            b2_accuracy = torch.sum(
                torch.round(bd_out_batch2[:, 0:1, :, :]) == bt_label_batch[:, 2:3, :, :]).item() / float(
                bd_out_batch2[:, 0:1, :, :].numel())
            d0_accuracy = torch.sum(
                torch.round(bd_out_batch0[:, 1:2, :, :]) == dire_label_batch_reg[:, 0:1, :, :]).item() / float(
                bd_out_batch0[:, 1:2, :, :].numel())
            d1_accuracy = torch.sum(
                torch.round(bd_out_batch1[:, 1:2, :, :]) == dire_label_batch_reg[:, 1:2, :, :]).item() / float(
                bd_out_batch1[:, 1:2, :, :].numel())
            d2_accuracy = torch.sum(
                torch.round(bd_out_batch2[:, 1:2, :, :]) == dire_label_batch_reg[:, 2:3, :, :]).item() / float(
                bd_out_batch2[:, 1:2, :, :].numel())

            q_L1_loss_list.append(q_L1_loss.item())
            b0_L1_loss_list.append(b0_L1_loss.item())
            b1_L1_loss_list.append(b1_L1_loss.item())
            b2_L1_loss_list.append(b2_L1_loss.item())
            d0_L1_loss_list.append(d0_L1_loss.item())
            d1_L1_loss_list.append(d1_L1_loss.item())
            d2_L1_loss_list.append(d2_L1_loss.item())

            q_accu_list.append(q_accuracy)
            b0_accu_list.append(b0_accuracy)
            b1_accu_list.append(b1_accuracy)
            b2_accu_list.append(b2_accuracy)
            d0_accu_list.append(d0_accuracy)
            d1_accu_list.append(d1_accuracy)
            d2_accu_list.append(d2_accuracy)

            del input_batch, qt_label_batch, bt_label_batch, dire_label_batch_reg, qt_out_batch, bd_out_batch0, bd_out_batch1, bd_out_batch2
    return [np.mean(q_L1_loss_list), np.mean(b0_L1_loss_list), np.mean(b1_L1_loss_list), np.mean(b2_L1_loss_list),
            np.mean(d0_L1_loss_list), np.mean(d1_L1_loss_list), np.mean(d2_L1_loss_list),
            np.mean(q_accu_list), np.mean(b0_accu_list), np.mean(b1_accu_list), np.mean(b2_accu_list),
            np.mean(d0_accu_list), np.mean(d1_accu_list), np.mean(d2_accu_list),
            np.mean(val_loss_list)]

@torch.no_grad()
def inference_pre_QBD(infe_loader_QB, Net_Q, Net_BD):  # for overall inference
    total_qt_out_batch = torch.zeros((1, 1, 8, 8))
    total_bt_out_batch = torch.zeros((1, 3, 16, 16))
    total_dire_out_batch_reg = torch.zeros((1, 3, 16, 16))
    with torch.no_grad():
        for step, data in enumerate(infe_loader_QB):
            # print("step: ", step)
            input_batch = data[0]
            input_batch = input_batch.cuda()
            qt_out_batch = Net_Q(input_batch)
            bd_out_batch0, bd_out_batch1, bd_out_batch2 = Net_BD(input_batch, qt_out_batch)
            bt_out_batch = torch.cat(
                [bd_out_batch0[:, 0:1, :, :], bd_out_batch1[:, 0:1, :, :], bd_out_batch2[:, 0:1, :, :]], 1)
            dire_out_batch = torch.cat(
                [bd_out_batch0[:, 1:2, :, :], bd_out_batch1[:, 1:2, :, :], bd_out_batch2[:, 1:2, :, :]], 1)
            # dire_out_batch = Net_D(input_batch, qt_out_batch, bt_out_batch)

            # qt_out_batch = torch.round(qt_out_batch).type(torch.int8)
            # bt_out_batch = torch.round(bt_out_batch).type(torch.int8)
            # dire_out_batch = torch.round(dire_out_batch).type(torch.int8)

            total_qt_out_batch = torch.cat([total_qt_out_batch, qt_out_batch.cpu()], 0)
            total_bt_out_batch = torch.cat([total_bt_out_batch, bt_out_batch.cpu()], 0)
            total_dire_out_batch_reg = torch.cat([total_dire_out_batch_reg, dire_out_batch.cpu()], 0)
            if step % 100 == 0:
                print("Number of finished blocks: ", total_qt_out_batch.shape[0])
            # del input_batch, qt_out_batch, bt_out_batch
        total_qt_out_batch = total_qt_out_batch[1:]
        total_bt_out_batch = total_bt_out_batch[1:]
        total_dire_out_batch_reg = total_dire_out_batch_reg[1:]

    return total_qt_out_batch, total_bt_out_batch, total_dire_out_batch_reg

@torch.no_grad()
def inference_pre_SepQBD(infe_loader_QB, Net_Q, Net_B, Net_D):  # for overall inference
    total_qt_out_batch = torch.zeros(1, 1, 8, 8)
    total_bt_out_batch = torch.zeros(1, 3, 16, 16)
    total_dire_out_batch_cla = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        for step, data in enumerate(infe_loader_QB):
            # print("step: ", step)
            input_batch = data[0]
            input_batch = input_batch.cuda()
            qt_out_batch = Net_Q(input_batch)
            bt_out_batch0, bt_out_batch1, bt_out_batch2 = Net_B(input_batch, qt_out_batch)
            bt_out_batch = torch.cat([bt_out_batch0, bt_out_batch1, bt_out_batch2], 1)
            dire_out_batch = Net_D(input_batch, qt_out_batch, bt_out_batch)
            num = dire_out_batch.shape[0]
            dire_out_batch_cla = torch.zeros(num, 3, 16, 16).cuda()
            for i in range(3):
                dire_out_batch_cla[:, i, :, :] = torch.argmax(dire_out_batch[:, i * 3:(i + 1) * 3, :, :], dim=1)

            total_qt_out_batch = torch.cat([total_qt_out_batch, qt_out_batch.cpu()], 0)
            total_bt_out_batch = torch.cat([total_bt_out_batch, bt_out_batch.cpu()], 0)
            total_dire_out_batch_cla = torch.cat([total_dire_out_batch_cla, dire_out_batch_cla.cpu()], 0)
            if step % 100 == 0:
                print("Number of finished blocks: ", total_qt_out_batch.shape[0])
            # del input_batch, qt_out_batch, bt_out_batch
        total_qt_out_batch = total_qt_out_batch[1:]
        total_bt_out_batch = total_bt_out_batch[1:]
        total_dire_out_batch_cla = total_dire_out_batch_cla[1:]

    return total_qt_out_batch, total_bt_out_batch, total_dire_out_batch_cla

#****************************************************************************************************************
# Joint Train
#****************************************************************************************************************

def Load_VP_Dataset(path, QP, batchSize, datasetID=0, isLuma=True, isQB=True, Net_QB=None):
    # [train validation test] [0 1 2] VVC partition
    if isLuma:
        comp = ['Luma', 'Y']
        block_size = '68'
    else:
        comp = ['Chroma', 'U', 'V']
        block_size = '34'
    tr_val_test = ['Train', 'Validate', 'TestSub']
    dataset_type = tr_val_test[datasetID]

    print('Start loading ' + comp[0] + ' ' + dataset_type + ' dataset...')
    input_path = os.path.join(path, dataset_type + '_' + comp[1] + '_Block' + block_size + '.npy')
    print('input path:', input_path)
    input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))
    if not isLuma:  # chroma input
        input_path1 = os.path.join(path, dataset_type + '_' + comp[2] + '_Block' + block_size + '.npy')
        print('input path1:', input_path1)
        input_batch1 = torch.FloatTensor(np.expand_dims(np.load(input_path1), 1))
        input_batch = torch.cat([input_batch, input_batch1], 1)  # concat U V component
        del input_batch1
    print('input_batch.shape:', input_batch.shape)

    if isQB:  # QB dataset
        qt_label_path = os.path.join(path, dataset_type + '_' + comp[0] + '_QP' + str(QP) + '_QTdepth_Block8.npy')
        bt_label_path = os.path.join(path, dataset_type + '_' + comp[0] + '_QP' + str(QP) + '_BTdepth_Block16.npy')
        print('qt_label path:', qt_label_path)
        print('bt_label path:', bt_label_path)
        qt_label_batch = torch.FloatTensor(np.expand_dims(np.load(qt_label_path), 1) - 1)  # qt depth start form 1
        bt_label_batch = torch.FloatTensor(np.expand_dims(np.load(bt_label_path), 1))
        # bt_label_batch += F.interpolate(qt_label_batch, scale_factor=2) * 2.0
        print('qt_label_batch.shape:', qt_label_batch.shape)
        print('bt_label_batch.shape:', bt_label_batch.shape)

        print("Creating QB data loader...")
        dataset = TensorDataset(input_batch, qt_label_batch, bt_label_batch)
        dataLoader = DataLoader(dataset=dataset, num_workers=2,
                                batch_size=batchSize,
                                pin_memory=True,
                                shuffle=True)
    else:  # D dataset
        print("Creating inference data loader...")
        infe_dataset = TensorDataset(input_batch)
        infe_loader = DataLoader(dataset=infe_dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=False)
        qt_out_batch, bt_out_batch = inference_QB(infe_loader, Net_QB)
        dire_label_path = os.path.join(path, dataset_type + '_' + comp[0] + '_QP' + str(QP) + '_MSdirection_Block16.npy')
        print('direction_label path:', dire_label_path)
        dire_label_batch_reg = torch.LongTensor(np.load(dire_label_path))
        dire_label_batch_cla = torch.LongTensor(
            torch.where(dire_label_batch_reg == -1, torch.full_like(dire_label_batch_reg, 2), dire_label_batch_reg))
        del dire_label_batch_reg
        print('qt_out_batch.shape:', qt_out_batch.shape)
        print('bt_out_batch.shape:', bt_out_batch.shape)
        print('dire_label_batch.shape:', dire_label_batch_cla.shape)

        print("Creating D data loader...")
        dataset = TensorDataset(input_batch, qt_out_batch, bt_out_batch, dire_label_batch_cla)
        dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=batchSize, pin_memory=True, shuffle=True)

    return dataLoader

@torch.no_grad()
def validation_BD(val_loader, Net_B, Net_D):
    bt_loss_list = []
    bt_L1_loss_list = []
    bt_accu_list = []
    dire_loss_list = []
    dire_L1_loss_list = []
    dire_accu_list = []

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            input_batch, qt_label_batch, bt_label_batch, dire_label_batch_cla = data
            input_batch = input_batch.cuda()
            qt_label_batch = qt_label_batch.cuda()
            bt_label_batch = bt_label_batch.cuda()
            dire_label_batch_cla = dire_label_batch_cla.cuda()

            bt_out_batch = Net_B(input_batch, qt_label_batch)
            dire_out_batch = Net_D(input_batch, qt_label_batch, bt_out_batch)

            dire_loss = loss_func_D(dire_out_batch, dire_label_batch_cla)
            num = dire_out_batch.shape[0]
            dire_out_batch_cla = torch.zeros(num, 3, 16, 16).cuda()
            for i in range(3):
                dire_out_batch_cla[:, i, :, :] = torch.argmax(dire_out_batch[:, i * 3:(i + 1) * 3, :, :], dim=1)
            dire_accuracy = torch.sum(dire_out_batch_cla == dire_label_batch_cla).item() / dire_out_batch_cla.numel()
            dire_L1_loss = L1_Loss(dire_out_batch_cla.float(), dire_label_batch_cla.float())

            bt_loss = Mul_Scale_L1Loss(bt_out_batch, bt_label_batch)
            bt_L1_loss = L1_Loss(bt_out_batch, bt_label_batch)
            bt_accuracy = torch.sum(torch.round(bt_out_batch) == bt_label_batch).item() / bt_out_batch.numel()

            bt_loss_list.append(bt_loss.item())
            bt_L1_loss_list.append(bt_L1_loss.item())
            bt_accu_list.append(bt_accuracy)

            dire_loss_list.append(dire_loss.item())
            dire_L1_loss_list.append(dire_L1_loss.item())
            dire_accu_list.append(dire_accuracy)

            del input_batch, qt_label_batch, bt_label_batch, dire_label_batch_cla, bt_out_batch, dire_out_batch, dire_out_batch
    return np.mean(bt_loss_list), np.mean(bt_L1_loss_list), np.mean(bt_accu_list), np.mean(dire_loss_list), np.mean(dire_L1_loss_list), np.mean(dire_accu_list)

@torch.no_grad()
def inference_QB(infe_loader_QB, Net_QB):  # for Net_D training
    total_qt_out_batch = torch.zeros(1, 1, 8, 8).cuda()
    total_bt_out_batch = torch.zeros(1, 1, 16, 16).cuda()
    with torch.no_grad():
        for step, data in enumerate(infe_loader_QB):
            input_batch = data[0]
            input_batch = input_batch.cuda()
            qt_out_batch, bt_out_batch = Net_QB(input_batch)
            total_qt_out_batch = torch.cat([total_qt_out_batch, qt_out_batch], 0)
            total_bt_out_batch = torch.cat([total_bt_out_batch, bt_out_batch], 0)
            # del input_batch, qt_out_batch, bt_out_batch
        total_qt_out_batch = total_qt_out_batch[1:]
        total_bt_out_batch = total_bt_out_batch[1:]

    return total_qt_out_batch, total_bt_out_batch

#****************************************************************************************************************
# Inference
#****************************************************************************************************************

def Load_Infe_VP_Dataset(pre_path, batchSize, isLuma=True):
    # [train validation test] [0 1 2] VVC partition
    if isLuma:
        comp = ['Luma', 'Y']
        block_size = '68'
    else:
        comp = ['Chroma', 'U', 'V']
        block_size = '34'
    tr_val_test = ['Train', 'Validate', 'TestSub']
    dataset_type = "Test"

    print('Start loading ' + comp[0] + ' ' + dataset_type + ' dataset...')
    input_path = pre_path + '_' + comp[1] + '_Block' + block_size + '.npy'
    print('input path:', input_path)
    input_batch = torch.FloatTensor(np.expand_dims(np.load(input_path), 1))[636231:636231+3458]
    if not isLuma:  # chroma input
        input_path1 = pre_path + '_' + comp[2] + '_Block' + block_size + '.npy'
        print('input path1:', input_path1)
        input_batch1 = torch.FloatTensor(np.expand_dims(np.load(input_path1), 1))
        input_batch = torch.cat([input_batch, input_batch1], 1)  # concat U V component
        del input_batch1
    print('input_batch.shape:', input_batch.shape)

    print("Creating inference data loader...")
    dataset = TensorDataset(input_batch)
    dataLoader = DataLoader(dataset=dataset, num_workers=0, batch_size=batchSize, pin_memory=True,  shuffle=False)

    return dataLoader

@torch.no_grad()
def inference_QBD(infe_loader_QB, Net_QB, Net_D):  # for overall inference
    total_qt_out_batch = torch.zeros(1, 1, 8, 8)
    total_bt_out_batch = torch.zeros(1, 3, 16, 16)
    total_dire_out_batch_cla = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        for step, data in enumerate(infe_loader_QB):
            # print("step: ", step)
            input_batch = data[0]
            input_batch = input_batch.cuda()
            qt_out_batch, bt_out_batch = Net_QB(input_batch)
            dire_out_batch = Net_D(input_batch, qt_out_batch, bt_out_batch)
            num = dire_out_batch.shape[0]
            dire_out_batch_cla = torch.zeros(num, 3, 16, 16).cuda()
            for i in range(3):
                dire_out_batch_cla[:, i, :, :] = torch.argmax(dire_out_batch[:, i * 3:(i + 1) * 3, :, :], dim=1)

            total_qt_out_batch = torch.cat([total_qt_out_batch, qt_out_batch.cpu()], 0)
            total_bt_out_batch = torch.cat([total_bt_out_batch, bt_out_batch.cpu()], 0)
            total_dire_out_batch_cla = torch.cat([total_dire_out_batch_cla, dire_out_batch_cla.cpu()], 0)
            print("Number of finished blocks: ", total_qt_out_batch.shape[0])
            # del input_batch, qt_out_batch, bt_out_batch
        total_qt_out_batch = total_qt_out_batch[1:]
        total_bt_out_batch = total_bt_out_batch[1:]
        total_dire_out_batch_cla = total_dire_out_batch_cla[1:]

    return total_qt_out_batch, total_bt_out_batch, total_dire_out_batch_cla

#****************************************************************************************************************
# Post Process Metrics
#****************************************************************************************************************
def check_square_unity(mat):  # input 4*4 tensor
    num0 = len(torch.where(mat == 0)[0])
    if num0 >= 0 and num0 <= 12:  # 0 in the minority
        mat = torch.where(mat == 0, torch.full_like(mat, 1).cuda(), mat)
        # process 4 sub-mats
        for i in [0, 2]:
            for j in [0, 2]:
                sum_sub_mat = torch.sum(mat[i:i + 2, j:j + 2])
                if sum_sub_mat <= 10 and sum_sub_mat >= 5: # 1 and 2 or 3 mixed
                    sub_num1 = len(torch.where(mat[i:i + 2, j:j + 2] == 1)[0])
                    if sub_num1 < 3:
                        mat[i:i + 2, j:j + 2] = torch.where(mat[i:i + 2, j:j + 2] == 1, (torch.ones((2, 2)) * 2).cuda(), mat[i:i + 2, j:j + 2])
                    else:
                        mat[i:i + 2, j:j + 2] = torch.ones((2, 2)).cuda()
    elif num0 > 12 and num0 < 16:
        mat = torch.zeros((4, 4)).cuda()
    return mat

def eli_structual_error(out_batch):
    N = out_batch.shape[0]
    pooled_batch = torch.clamp(torch.round(F.max_pool2d(out_batch, 2)), min=0, max=3)
    for num in range(N):
        pooled_batch[num][0] = check_square_unity(pooled_batch[num][0])
    post_batch = F.interpolate(pooled_batch, scale_factor=2)
    del pooled_batch
    return post_batch

def get_norm_block(depth, x, y, norm_block, qt_map, block_size):
    cur_depth = qt_map[x, y]

    if cur_depth == depth:  # end partition
        sub_size = block_size >> depth
        scale = block_size // 8
        block_x = x * scale
        block_y = y * scale
        block_mean = torch.mean(norm_block[block_x:block_x+sub_size, block_y:block_y+sub_size])
        # block_std = torch.std(norm_block[block_x:block_x+sub_size, block_y:block_y+sub_size])
        # if block_std == 0:
        #     block_std = 1
        # normalize
        norm_block[block_x:block_x+sub_size, block_y:block_y+sub_size] -= block_mean
        return
    elif cur_depth > depth:  # carry on partition
        sub_map_size = 8 >> depth
        for i_offset in range(2):
            for j_offset in range(2):
                get_norm_block(depth + 1, x + i_offset * sub_map_size // 2, y + j_offset * sub_map_size // 2, norm_block, qt_map, block_size)
    return

# normalize the input block according to qt map
def block_qtnode_norm(qt_map, block, isLuma=True):
    b, c, h, w = block.shape
    if isLuma:
        block_size = 64
    else:
        block_size = 32
    # post_qt_map = eli_structual_error(qt_map)
    post_qt_map = torch.clamp(torch.round(qt_map), min=0, max=3).cuda()
    norm_block = torch.FloatTensor(b, c, block_size, block_size).cuda()
    norm_block[:, :, :, :] = block[:, :, h-block_size:h, w-block_size:w].detach()
    for i in range(b):
        for j in range(c):
            get_norm_block(0, 0, 0, norm_block[i][j], post_qt_map[i][0], block_size)
    del post_qt_map
    return Variable(norm_block, requires_grad=False)

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain_model(current_model, pretrain_model):
    source_dict = torch.load(pretrain_model)
    if "state_dict" in source_dict.keys():
        source_dict = remove_prefix(source_dict['state_dict'], 'module.')
    else:
        source_dict = remove_prefix(source_dict, 'module.')
    dest_dict = current_model.state_dict()
    trained_dict = {k: v for k, v in source_dict.items() if k in dest_dict and source_dict[k].shape == dest_dict[k].shape}
    dest_dict.update(trained_dict)
    current_model.load_state_dict(dest_dict)
    # for k, v in trained_dict.items():
    #     if "conv_d2.bias" in k:
    #         print(k)
    #         print(v)
    # for k, v in dest_dict.items():
    #     if "conv_d2.bias" in k:
    #         print(k)
    #         print(v)
    return current_model


def load_sequences_info():
    num = 22
    seqs_info_path = r"E:\VVC-Fast-Partition-DP\Code\Debug\VVC_Test_Sequences.txt"
    seqs_info_fp = open(seqs_info_path, 'r')
    data = []
    for line in seqs_info_fp:
        if "end!!!!" in line:
            break
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:num, 0]
    seqs_path_name = data[:num, 1]
    seqs_width = data[:num, 2].astype(np.int64)  # enough bits for calculating h*w
    seqs_height = data[:num, 3].astype(np.int64)
    seqs_frmnum = data[:num, 4].astype(np.int64)
    sub_frmnum_list, block_num_list = [], []
    for i in range(num):
        SubSampleRatio = 30
        if i >= 79:
            SubSampleRatio = 1
        SubSampleRatio = 8
        sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        sub_frmnum_list.append(sub_frmnum)
        block_num = (seqs_width[i] // 64) * (seqs_height[i] // 64) * sub_frmnum
        block_num_list.append(block_num)

    return seqs_path_name, seqs_width, seqs_height, sub_frmnum_list, block_num_list

def post_process(qt_out_batch, bt_out_batch, dire_out_batch, comp, qp, save_dir):
    if comp == "Luma":
        is_luma = True
    else:
        is_luma = False
    qt_out_batch = eli_structual_error(qt_out_batch).cpu().numpy().squeeze(axis=1)
    # dire_out_batch_cla = dire_out_batch_cla.cpu().numpy()
    start_block_id = 0
    seqs_path_name, seqs_width, seqs_height, sub_frmnum_list, block_num_list = load_sequences_info()
    for seq_id in range(0, 22):
        seq_name = seqs_path_name[seq_id].rstrip(".yuv")
        width = seqs_width[seq_id]
        height = seqs_height[seq_id]
        sub_frmnum = sub_frmnum_list[seq_id]
        block_num = block_num_list[seq_id]
        print(comp, qp, seq_name)
        input_qt_batch = qt_out_batch[start_block_id:start_block_id + block_num]
        input_bt_batch = bt_out_batch[start_block_id:start_block_id + block_num]
        input_dire_batch = dire_out_batch[start_block_id:start_block_id + block_num]
        start_block_id += block_num

        save_path = os.path.join(save_dir, seq_name + "_" + comp + "_QP" + str(qp) + "_PartitionMat.txt")
        print("Save:", save_path)
        get_sequence_partition_for_VTM(qt_map=input_qt_batch, bt_map=input_bt_batch, dire_map=input_dire_batch,
                                       is_luma=is_luma,
                                       save_path=save_path, frm_num=sub_frmnum, frm_width=width, frm_height=height)


    del qt_out_batch
