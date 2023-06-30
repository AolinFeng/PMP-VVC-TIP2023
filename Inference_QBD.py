'''
Function:
  Network inference + Post processing

Main functions:
  * inference_VVC(args)

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
import time

import Model_QBD as model
from Metrics import Load_Infe_VP_Dataset, inference_pre_QBD, post_process

SAVE_MID_RESULT = False
POST_PROCESS = True


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
    trained_dict = {k: v for k, v in source_dict.items() if
                    k in dest_dict and source_dict[k].shape == dest_dict[k].shape}
    dest_dict.update(trained_dict)
    current_model.load_state_dict(dest_dict)
    # for k, v in trained_dict.items():
    #     print(k)
    return current_model


def load_sequences_info():
    num = 22
    seqs_info_path = "VVC_Test_Sequences.txt"
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
    sub_frmnum_list = []
    for i in range(num):
        SubSampleRatio = 30
        if i >= 79:
            SubSampleRatio = 1
        SubSampleRatio = 8
        sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        sub_frmnum_list.append(sub_frmnum)
    return seqs_path_name, seqs_width, seqs_height, sub_frmnum_list


@torch.no_grad()
def inference_VVC(args):
    qp_base_list = [22, 27, 32, 37]
    qp_list = []
    for qp_id in range(args.startQPID, args.startQPID + args.qpNum):
        qp_list.append(qp_base_list[qp_id])
    print("QP:", qp_list)
    net_time = 0
    post_time = 0
    # seqs_path_name, seqs_width, seqs_height, sub_frmnum_list = load_sequences_info()
    # seqs_num = 22
    # comp_list = ["Luma", "Chroma"]
    if POST_PROCESS:
        save_dir = os.path.join(args.outDir, args.jobID, 'PartitionMat')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if SAVE_MID_RESULT:
        save_mid_dir = os.path.join(args.outDir, args.jobID, 'OutputMap')
        if not os.path.exists(save_mid_dir):
            os.makedirs(save_mid_dir)

    data_name = "Test"
    # input_pre_path = os.path.join(args.inputDir, data_name)
    for comp in ["Luma", "Chroma"]:
        is_luma = False
        max_bt_depth = 4
        if comp == "Luma":
            is_luma = True
            max_bt_depth = 5
        QB_test_loader = Load_Infe_VP_Dataset(pre_path=args.inputDir, batchSize=args.batchSize, isLuma=is_luma)
        for qp in qp_list:
            print(comp + " QP" + str(qp) + " network inference start...")
            net_start_time = time.time()
            ################################################### Load Models #######################################################
            if comp == "Luma":
                Net_Q = model.Luma_Q_Net()
                Net_BD = model.Luma_MSBD_Net()
                # Net_D = model.Luma_D_Net()
            else:
                Net_Q = model.Chroma_Q_Net()
                Net_BD = model.Chroma_MSBD_Net()
                # Net_D = model.Chroma_D_Net()

            net_Q_path = "./Models/" + comp + "_Q_" + str(qp) + ".pkl"
            net_BD_path = "./Models/" + comp + "_BD_" + str(qp) + ".pkl"
            # net_D_path = "./Models/" + comp + "_D_" + str(qp) + ".pkl"
            # Net_D.load_state_dict(torch.load(net_D_path))
            # print("Start loading network")
            Net_Q = load_pretrain_model(Net_Q, net_Q_path)
            Net_BD = load_pretrain_model(Net_BD, net_BD_path)
            Net_Q = nn.DataParallel(Net_Q).cuda()
            Net_BD = nn.DataParallel(Net_BD).cuda()

            ################################################### Network Inference #######################################################
            qt_out_batch, bt_out_batch, dire_out_batch_reg = inference_pre_QBD(QB_test_loader, Net_Q, Net_BD)
            net_time += time.time() - net_start_time

            if SAVE_MID_RESULT:
                save_qt_path = os.path.join(save_mid_dir, data_name + '_' + comp + '_QP' + str(qp) + "_QTdepth.npy")
                save_bt_path = os.path.join(save_mid_dir, data_name + '_' + comp + '_QP' + str(qp) + "_MSBTdepth.npy")
                save_dire_path = os.path.join(save_mid_dir,
                                              data_name + '_' + comp + '_QP' + str(qp) + "_MSdirection.npy")
                np.save(save_qt_path, qt_out_batch.cpu().numpy())
                np.save(save_bt_path, bt_out_batch.cpu().numpy())
                np.save(save_dire_path, dire_out_batch_reg.cpu().numpy())

            ############################################### Post Process for Each Sequence ####`###########################################
            if POST_PROCESS:
                print(comp + " QP" + str(qp) + " post process start...")
                post_start_time = time.time()
                qt_out_batch = torch.FloatTensor(qt_out_batch).cuda()  # b*1*8*8
                # bt_out_batch = torch.FloatTensor(bt_out_batch)  # b*3*16*16
                # bt_out_batch = torch.clamp(torch.round(bt_out_batch), min=0, max=max_bt_depth).cpu().numpy()
                # dire_out_batch_reg = torch.FloatTensor(dire_out_batch_reg).cpu()
                # dire_out_batch_reg = torch.clamp(torch.round(dire_out_batch_reg), min=-1, max=1).numpy()
                # dire_out_batch_cla = torch.LongTensor(
                #     torch.where(dire_out_batch_reg == -1, torch.full_like(dire_out_batch_reg, 2), dire_out_batch_reg)).numpy()
                bt_out_batch = bt_out_batch.cpu().numpy()
                dire_out_batch_reg = dire_out_batch_reg.cpu().numpy()
                # bt_out_batch[:, 1:2, :, :] = bt_out_batch[:, 1:2, :, :] + bt_out_batch[:, 0:1, :, :]
                # bt_out_batch[:, 2:3, :, :] = bt_out_batch[:, 2:3, :, :] + bt_out_batch[:, 1:2, :, :]
                post_process(qt_out_batch, bt_out_batch, dire_out_batch_reg, comp, qp, save_dir)
                post_time += time.time() - post_start_time

    print("Network time: ", net_time)
    print("Post process time: ", post_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default='/output/')
    # parser.add_argument('--log_dir', type=str, default='/output/log')
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--startQPID', default=0, type=int, help='QP start ID')
    parser.add_argument('--qpNum', default=1, type=int, help='test QP number')

    args = parser.parse_args()

    start_time = time.time()
    inference_VVC(args)
    infe_time = time.time() - start_time
    print('Total inference time:', infe_time)
    '''
    python /code/DebugInference_QBD.py --batchSize 400 --inputDir /data/FengAolin/VTM10_Partition_LMCS0_LFNST0 --outDir /output/
    python dp_inference.py --input_dir /DataSet --batchSize 200
    startdocker -P /ghome/fengal -D /gdata/fengal -c "python /ghome/fengal/HM_Fast_Partition/dp_train.py --input_dir /gdata/fengal/HM_Fast_Partition --out_dir /output/" bit:5000/wangyc-pytorch1.0.1_cuda10.0_apex
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    '''




