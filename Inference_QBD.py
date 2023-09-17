'''
Function:
  Network inference + Post processing

Main functions:
  * inference_VVC_seqs(args)

Author: Aolin Feng
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time

import Model_QBD as model
from Metrics import inference_pre_QBD, post_process, seq_post_process

SAVE_MID_RESULT = False
POST_PROCESS = True
SSRatio = 30

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
    num = 79
    seqs_info_path = r"Training_Sequences.txt" # VVC_Test_Sequences
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
        SubSampleRatio = SSRatio
        sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        sub_frmnum_list.append(sub_frmnum)
        block_num = (seqs_width[i] // 64) * (seqs_height[i] // 64) * sub_frmnum
        block_num_list.append(block_num)

    return seqs_name, seqs_path_name, seqs_width, seqs_height, seqs_frmnum, sub_frmnum_list, block_num_list

def import_yuv420(file_path, width, height, frm_num, SubSampleRatio=1, is10bit=False):
    fp = open(file_path,'rb')
    pixnum = width * height
    subnumfrm = (frm_num + SubSampleRatio - 1) // SubSampleRatio # actual frame number after downsampling
    if is10bit:
        data_type = np.uint16
    else:
        data_type = np.uint8
    y_temp = np.zeros(pixnum*subnumfrm, dtype=data_type)
    u_temp = np.zeros(pixnum*subnumfrm // 4, dtype=data_type)
    v_temp = np.zeros(pixnum*subnumfrm // 4, dtype=data_type)
    for i in range(0, frm_num, SubSampleRatio):
        if is10bit:
            fp.seek(i * pixnum * 3, 0)
        else:
            fp.seek(i * pixnum * 3 // 2, 0)
        subi = i // SubSampleRatio
        y_temp[subi*pixnum : (subi+1)*pixnum] = np.fromfile(fp, dtype=data_type, count=pixnum, sep='')
        u_temp[subi*pixnum//4 : (subi+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
        v_temp[subi*pixnum//4 : (subi+1)*pixnum//4] = np.fromfile(fp, dtype=data_type, count=pixnum//4, sep='')
    fp.close()
    y = y_temp.reshape((subnumfrm, height, width))
    u = u_temp.reshape((subnumfrm, height//2, width//2))
    v = v_temp.reshape((subnumfrm, height//2, width//2))
    return y, u, v  # return frm_num * H * W

def output_block_yuv(file_path, width, height, block_size, in_overlap, numfrm, SubSampleRatio, is10bit=False, save_path=None):
    y, u, v = import_yuv420(file_path, width, height, numfrm, SubSampleRatio, is10bit=is10bit)
    if is10bit:
        y = (np.round(y / 4)).clip(0, 255).astype(np.uint8)
        u = (np.round(u / 4)).clip(0, 255).astype(np.uint8)
        v = (np.round(v / 4)).clip(0, 255).astype(np.uint8)
    block_num_in_width = width // block_size
    block_num_in_height = height // block_size
    # print(block_num_in_width, block_num_in_height)
    for id, comp in enumerate([y, u, v]):
        if id == 0:
            overlap = in_overlap
            comp_block_size = block_size
        else:
            overlap = int(in_overlap / 2)
            comp_block_size = block_size // 2
        pad_comp = np.zeros((comp.shape[0], comp.shape[1]+overlap, comp.shape[2]+overlap), dtype=np.uint8)
        pad_comp[:, overlap:, overlap:] = comp
        subnumfrm = comp.shape[0]

        block_list = []
        for f_num in range(subnumfrm):
            for i in range(block_num_in_height):
                for j in range(block_num_in_width):
                    block_list.append(pad_comp
                         [f_num, i * comp_block_size:(i + 1) * comp_block_size + overlap, j * comp_block_size:(j + 1) * comp_block_size + overlap])
        if id == 0:
            block_y = np.array(block_list)
        elif id == 1:
            block_u = np.array(block_list)
        else:
            block_v = np.array(block_list)

    if save_path is not None:
        out_fp = open(save_path, "wb")
        for i in range(block_y.shape[0]):
            out_fp.write(block_y[i].reshape(-1))
            out_fp.write(block_u[i].reshape(-1))
            out_fp.write(block_v[i].reshape(-1))
        out_fp.close()

    # print('shape of block_y', block_y.shape)
    # print('shape of block_u', block_u.shape)
    # print('shape of block_v', block_v.shape)
    # del block_y, block_u, block_v
    return block_y, block_u, block_v  # num_block * block_size * block_size

@torch.no_grad()
def inference_VVC_seqs(args):
    save_dir = os.path.join(args.outDir, args.jobID, "PartitionMat")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seqs_block_time = np.zeros(22)
    seqs_net_time = np.zeros((22, 4, 2))
    seqs_post_time = np.zeros((22, 4, 2))

    seqs_name, seqs_path_name, seqs_width, seqs_height, seqs_frmnum, sub_frmnum_list, block_num_list = load_sequences_info()
    seq_cfg_dir = r".\per-sequence"
    for seq_id in range(args.startSeqID, args.startSeqID + args.seqNum):
        # ********************************** Load Sequence Information *************************************
        seq_name = seqs_name[seq_id]
        seq_path_name = seqs_path_name[seq_id].rstrip(".yuv")
        width = seqs_width[seq_id]
        height = seqs_height[seq_id]
        numfrm = seqs_frmnum[seq_id]
        sub_numfrm = sub_frmnum_list[seq_id]
        block_num = block_num_list[seq_id]
        is10bit = False
        seq_cfg_path = os.path.join(seq_cfg_dir, seq_name + ".cfg")
        seq_cfg_fp = open(seq_cfg_path)
        for line in seq_cfg_fp:
            if "InputFile" in line:
                line = line.rstrip("\n").split('#')[0]  # remove annotation
                line = line.replace(" ", "")   # remove space
                seq_path = line.split(":", 1)[1]  # sequence path

            elif "InputBitDepth" in line:
                line = line.rstrip("\n").split('#')[0]  # remove annotation
                line = line.replace(" ", "")   # remove space
                bit_depth = line.split(":", 1)[1]
                if bit_depth == "10":
                    is10bit = True
        print(seq_name)
        # ********************************** Load Input Blocks *************************************
        start_time = time.time()
        block_y, block_u, block_v = output_block_yuv(seq_path, width, height, block_size=64, in_overlap=4,
                                                     numfrm=numfrm, SubSampleRatio=SSRatio, is10bit=is10bit)
        seqs_block_time[seq_id-args.startSeqID] = time.time() - start_time

        for comp_id, comp in enumerate(["Luma", "Chroma"]):
            input_batch = torch.FloatTensor(np.expand_dims(block_y, 1))
            if comp == "Chroma":
                input_batch = F.max_pool2d(input_batch, 2)
                input_batch1 = torch.FloatTensor(np.expand_dims(block_u, 1))
                input_batch2 = torch.FloatTensor(np.expand_dims(block_v, 1))
                input_batch = torch.cat([input_batch, input_batch1, input_batch2], 1)
                del input_batch1, input_batch2
            # print('input_batch.shape:', input_batch.shape)

            # print("Creating inference data loader...")
            dataset = TensorDataset(input_batch)
            QB_test_loader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=False)

            for qp in [22, 27, 32, 37]:
                # ********************************** Load Models *************************************
                start_time = time.time()
                if comp == "Luma":
                    Net_Q = model.Luma_Q_Net()
                    Net_BD = model.Luma_MSBD_Net()
                    # Net_D = model.Luma_D_Net()
                else:
                    Net_Q = model.Chroma_Q_Net()
                    Net_BD = model.Chroma_MSBD_Net()

                net_Q_path = "./CTU_Models/" + comp + "_Q_" + str(qp) + ".pkl"
                net_BD_path = "./CTU_Models/" + comp + "_BD_" + str(qp) + ".pkl"
                Net_Q = load_pretrain_model(Net_Q, net_Q_path)
                Net_BD = load_pretrain_model(Net_BD, net_BD_path)
                Net_Q = nn.DataParallel(Net_Q).cuda()
                Net_BD = nn.DataParallel(Net_BD).cuda()
                # ********************************** Network Inference *************************************
                qt_out_batch, bt_out_batch, dire_out_batch_reg = inference_pre_QBD(QB_test_loader, Net_Q, Net_BD)
                seqs_net_time[seq_id-args.startSeqID, (qp-22)//5, comp_id] = time.time() - start_time

                # ********************************** Post Process ************************************
                start_time = time.time()
                qt_out_batch = torch.FloatTensor(qt_out_batch).cuda()  # b*1*8*8
                bt_out_batch = bt_out_batch.cpu().numpy()
                dire_out_batch_reg = dire_out_batch_reg.cpu().numpy()
                # bt_out_batch[:, 1:2, :, :] = bt_out_batch[:, 1:2, :, :] + bt_out_batch[:, 0:1, :, :]
                # bt_out_batch[:, 2:3, :, :] = bt_out_batch[:, 2:3, :, :] + bt_out_batch[:, 1:2, :, :]

                save_path = os.path.join(save_dir, seq_path_name + "_" + comp + "_QP" + str(qp) + "_PartitionMat.txt")
                print("Save:", save_path)
                seq_post_process(qt_out_batch, bt_out_batch, dire_out_batch_reg, comp, sub_numfrm, width, height, save_path)

                seqs_post_time[seq_id-args.startSeqID, (qp-22)//5, comp_id] = time.time() - start_time
    # ********************************** Log Time Information ************************************
    sta_log_path = os.path.join(args.outDir, args.jobID,
                                "Time_Sta_" + str(args.startSeqID) + "_" + str(args.startSeqID + args.seqNum) + ".txt")
    sta_log_fp = open(sta_log_path, "w")
    for seq_id in range(args.seqNum):
        for qp_id in range(4):
            for s in [str(seqs_block_time[seq_id]),
                      str(seqs_net_time[seq_id, qp_id, 0]), str(seqs_net_time[seq_id, qp_id, 1]),
                      str(seqs_post_time[seq_id, qp_id, 0]), str(seqs_post_time[seq_id, qp_id, 1])]:
                sta_log_fp.write(s)
                sta_log_fp.write(',')
            sta_log_fp.write('\n')

    print("Sum time:", np.sum(seqs_block_time) + np.sum(seqs_net_time) + np.sum(seqs_post_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobID', type=str, default='0000')
    parser.add_argument('--inputDir', type=str, default='/input/')
    parser.add_argument('--outDir', type=str, default='/output/')
    # parser.add_argument('--log_dir', type=str, default='/output/log')
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    parser.add_argument('--startSeqID', default=0, type=int, help='QP start ID')
    parser.add_argument('--seqNum', default=22, type=int, help='test QP number')

    args = parser.parse_args()

    start_time = time.time()
    inference_VVC_seqs(args)
    infe_time = time.time() - start_time
    print('Total inference time:', infe_time)
    '''
    python /code/DebugInference_QBD.py --batchSize 400 --inputDir /data/FengAolin/VTM10_Partition_LMCS0_LFNST0 --outDir /output/
    python dp_inference.py --input_dir /DataSet --batchSize 200
    startdocker -P /ghome/fengal -D /gdata/fengal -c "python /ghome/fengal/HM_Fast_Partition/dp_train.py --input_dir /gdata/fengal/HM_Fast_Partition --out_dir /output/" bit:5000/wangyc-pytorch1.0.1_cuda10.0_apex
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    --startQPID 2 --qpNum 1 --batchSize 400 --inputDir E:\VVC-Fast-Partition-DP\Dataset\Input --outDir E:\VVC-Fast-Partition-DP\Output\Test
    '''




