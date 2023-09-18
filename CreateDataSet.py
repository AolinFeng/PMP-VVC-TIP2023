'''
Function:
  Create dataset
  From [binary yuv frame] to [blocks of individual yuv channels]
  From [partition information saved from encoder] to [incomplete partition map]

Main functions:
  * save_sequence_block_set(): partition the three channels of input yuv file into numpy array. (input of the network)
  * save_partition_block_set(): using partition information saved from the encoder to get incomplete partition map. (label of the network)
  * concate_seqs(): concatenate the individual yuv block files (generated from different yuv sequences) to be used for training
  * concate_partition(): similar as concate_seqs(), but for partition map

Note:
  * The reason it's incomplete is that this code only outputs the qt depth map, direction map, and the last layer of bt depth map.
    The multi-layer bt depth map needs the code "GenMSBTMap.py" to get.
  * To get "partition information saved from the encoder:, reference the Macro definition "Save_Depth_fal" in the VTM codec.
  * Functions beyond the four main functions serve as reference.

Author: Aolin Feng
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import time

# from yuv420 binary file to numpy array
def import_yuv420(file_path, width, height, frm_num, SubSampleRatio=1, show=False, is10bit=False):
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
    if show:
        for i in range(subnumfrm):
            print(i)
            plt.imshow(y[i, :, :], cmap='gray')
            plt.show()
            plt.pause(1)
    return y, u, v  # return frm_num * H * W

def clip_yuv(filename, width, height, startfrm, endfrm):
    fp = open(filename,'rb')
    frmsize = width * height * 3 // 2
    start_loc = startfrm * frmsize
    read_size = (endfrm - startfrm + 1) * frmsize
    fp.seek(start_loc, 0)
    data = fp.read(read_size)
    fp.close()
    out_fp = open('out_clip_yuv.yuv', 'wb')
    out_fp.write(data)
    out_fp.close()

def cut_yuv(filename, width, height, block_size, numfrm):
    width_cut = int(width // block_size * block_size)
    height_cut = int(height // block_size * block_size)
    y, u, v = import_yuv420(filename, width, height, numfrm)
    out_fp = open('out_cut_yuv.yuv', 'ab')
    for i in range(numfrm):
        y_cut = y[i, 0:height_cut, 0:width_cut]
        u_cut = u[i, 0:height_cut//2, 0:width_cut//2]
        v_cut = v[i, 0:height_cut//2, 0:width_cut//2]
        print(i, ' ', y_cut.shape)
        out_fp.write(y_cut.reshape(-1))
        out_fp.write(u_cut.reshape(-1))
        out_fp.write(v_cut.reshape(-1))
    out_fp.close()

# save blocks of y component (with overlap)
def output_block_yuv(file_path, width, height, block_size, in_overlap, numfrm, SubSampleRatio, is10bit=False, save_path=None):
    y, u, v = import_yuv420(file_path, width, height, numfrm, SubSampleRatio, is10bit=is10bit)
    if is10bit:
        y = (np.round(y / 4)).clip(0, 255).astype(np.uint8)
        u = (np.round(u / 4)).clip(0, 255).astype(np.uint8)
        v = (np.round(v / 4)).clip(0, 255).astype(np.uint8)
    block_num_in_width = width // block_size
    block_num_in_height = height // block_size
    print(block_num_in_width, block_num_in_height)
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

    print('shape of block_y', block_y.shape)
    print('shape of block_u', block_u.shape)
    print('shape of block_v', block_v.shape)
    check_id = 29
    plt.figure(0)
    plt.imshow(block_v[check_id], cmap='gray')
    # plt.figure(1)
    # plt.imshow(block_u[check_id], cmap='gray')
    # plt.figure(2)
    # plt.imshow(block_v[check_id], cmap='gray')
    plt.show()
    return block_y, block_u, block_v  # num_block * block_size * block_size

def save_sequence_block_set():
    seqs_info_path = r'.\Cfg\VVC_Test_Sequences.txt'
    seqs_root_dir = r'E:\Research\testyuv\VVC'
    save_dir = r'E:\VVC-Fast-Partition-DP\Dataset\Test'
    seqs_info_fp = open(seqs_info_path, 'r')
    # load input sequence information
    data = []
    for line in seqs_info_fp:
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:, 0]
    seqs_path_name = data[:, 1]
    seqs_width = data[:, 2].astype(np.int64)  # enough bits for calculating h*w
    seqs_height = data[:, 3].astype(np.int64)
    seqs_frmnum = data[:, 4].astype(np.int64)
    num_seq = 85
    for i_seq in range(0, 22):
        path = os.path.join(seqs_root_dir, seqs_path_name[i_seq])
        print(path, i_seq)
        width = seqs_width[i_seq]
        height = seqs_height[i_seq]
        frmnum = seqs_frmnum[i_seq]
        # print(width,height,frmnum)
        if i_seq < 8:
            is10bit = True
        else:
            is10bit = False
        block_y, block_u, block_v = output_block_yuv(path, width, height, block_size=64, in_overlap=4, numfrm=frmnum, SubSampleRatio=8, is10bit=is10bit)
        save_name_y = seqs_name[i_seq] + '_Y_Block68.npy'
        save_name_u = seqs_name[i_seq] + '_U_Block34.npy'
        save_name_v = seqs_name[i_seq] + '_V_Block34.npy'
        save_path_y = os.path.join(save_dir + r'\Block_Y', save_name_y)
        save_path_u = os.path.join(save_dir + r'\Block_U', save_name_u)
        save_path_v = os.path.join(save_dir + r'\Block_V', save_name_v)
        print(save_path_y)
        print(save_path_u)
        print(save_path_v)
        np.save(save_path_y, block_y)
        np.save(save_path_u, block_u)
        np.save(save_path_v, block_v)

# output blocks of qt depth map, mt depth map and direction map from saved information.
# you can infer the information thta the encoder needs to save from this function.
def output_block_partition_map(file_path, frm_width, frm_height, frm_num, block_size=64, isChroma=False):
    depth_fp = open(file_path, 'r')
    qtdepth_mat = np.zeros((frm_num, frm_height // 4, frm_width // 4), dtype=np.uint8)
    btdepth_mat = np.zeros((frm_num, frm_height // 4, frm_width // 4), dtype=np.uint8)
    msdirection_mat = np.zeros((frm_num, 3, frm_height // 4, frm_width // 4), dtype=np.int8)
    frm_id = -1
    for line in depth_fp:
        # x y h w depth qtdepth btdepth mtdepth splitmode
        if 'frame' in line:
            frm_id += 1
            continue
        if isChroma:
            factor = 2
        else:
            factor = 1
        x = int(line.split(' ')[0]) * factor
        y = int(line.split(' ')[1]) * factor
        h = int(line.split(' ')[2]) * factor
        w = int(line.split(' ')[3]) * factor
        depth = int(line.split(' ')[4])
        qtdepth = int(line.split(' ')[5])
        btdepth = int(line.split(' ')[6])
        qtdepth_mat[frm_id, y // 4:(y + h) // 4, x // 4:(x + w) // 4] = qtdepth
        btdepth_mat[frm_id, y // 4:(y + h) // 4, x // 4:(x + w) // 4] = btdepth

        direction = 0
        for i in range(3):
            splitmode = int(line.split(' ')[8 + qtdepth + i])
            if splitmode == 2 or splitmode == 4:  # bth or tth
                direction = 1
            elif splitmode == 3 or splitmode == 5:  # btv or ttv
                direction = -1
            elif splitmode == 2000:
                direction = 0
            else:
                print('Error!!')
            msdirection_mat[frm_id, i, y // 4:(y + h) // 4, x // 4:(x + w) // 4] = direction

    qtdepth_mat = qtdepth_mat[:, ::2, ::2]  # down sample
    # print(qtdepth_mat.shape)
    # print(btdepth_mat.shape)
    # print(direction_mat.shape)
    # bt_plus = qtdepth_mat * 2 + btdepth_mat
    # plt.imshow(bt_plus[0], cmap='gray')
    # plt.axis('off')
    # plt.savefig('bt_plusqt.png', dpi=320, bbox_inches='tight', pad_inches=0)
    # plt.show()
    # save depth blocks
    block_num_in_width = frm_width // block_size
    block_num_in_height = frm_height // block_size
    qd_block_list = []
    bd_block_list = []
    msdire_block_list = []
    qd_block_size = block_size // 8
    md_block_size = block_size // 4
    dire_block_size = block_size // 4
    for f_num in range(frm_num):
        for i in range(block_num_in_height):
            for j in range(block_num_in_width):
                qd_block_list.append(qtdepth_mat[f_num, i * qd_block_size:(i + 1) * qd_block_size,
                                     j * qd_block_size:(j + 1) * qd_block_size])
                bd_block_list.append(btdepth_mat[f_num, i * md_block_size:(i + 1) * md_block_size,
                                     j * md_block_size:(j + 1) * md_block_size])
                msdire_block_list.append(msdirection_mat[f_num, :, i * dire_block_size:(i + 1) * dire_block_size,
                                       j * dire_block_size:(j + 1) * dire_block_size])
    qtdepth_block = np.array(qd_block_list)
    btdepth_block = np.array(bd_block_list)
    msdirection_block = np.array(msdire_block_list)
    del qtdepth_mat, btdepth_mat, msdirection_mat
    print(qtdepth_block.shape)
    print(btdepth_block.shape)
    print(msdirection_block.shape)
    # check_id = 104
    # print(qtdepth_block[check_id])
    # print(mtdepth_block[check_id])
    # print(direction_block[check_id])
    return qtdepth_block, btdepth_block, msdirection_block

def save_partition_block_set():
    comp = 'Chroma'
    is_chroma = False
    if comp == 'Chroma':
        is_chroma = True

    seqs_info_path = r'.\Cfg\Training_Sequences.txt'
    # seqs_root_dir = r'E:\Research\testyuv\Dataset\Train'
    partition_root_dir = r'E:\VVC-Fast-Partition-DP\Dataset\Partition_Info\TestSeq\DepthSaving'
    save_dir = r'E:\VVC-Fast-Partition-DP\Dataset'
    seqs_info_fp = open(seqs_info_path, 'r')
    data = []
    for line in seqs_info_fp:
        if "end!!!!" in line:
            break
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:, 0]
    seqs_path_name = data[:, 1]
    seqs_width = data[:, 2].astype(np.int64)  # enough bits for calculating h*w
    seqs_height = data[:, 3].astype(np.int64)
    seqs_frmnum = data[:, 4].astype(np.int64)
    sub_frmnum_list = []
    for i in range(91):
        SubSampleRatio = 8
        sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        sub_frmnum_list.append(sub_frmnum)

    for qp in [22, 27, 32, 37]:
        concate_qt_block = np.zeros((1, 8, 8), dtype=np.uint8)
        concate_bt_block = np.zeros((1, 16, 16), dtype=np.uint8)
        concate_direction_block = np.zeros((1, 3, 16, 16), dtype=np.int8)
        for i_seq in range(88, 91):
            partition_path = os.path.join(partition_root_dir, seqs_name[i_seq] + '_QP' + str(qp) + '_' + comp + '_Partition.txt')
            print(partition_path)
            width = seqs_width[i_seq]
            height = seqs_height[i_seq]
            sub_frmnum = sub_frmnum_list[i_seq]
            qtdepth_block, btdepth_block, msdirection_block = output_block_partition_map(file_path=partition_path,
                                                                                         frm_width=width,
                                                                                         frm_height=height,
                                                                                         frm_num=sub_frmnum,
                                                                                         block_size=64,
                                                                                         isChroma=is_chroma)
            block_num = qtdepth_block.shape[0]
            if block_num < 5000:
                sub_num = 500
            else:
                sub_num = 5000
            concate_qt_block = np.concatenate((concate_qt_block, qtdepth_block), axis=0)
            concate_bt_block = np.concatenate((concate_bt_block, btdepth_block), axis=0)
            concate_direction_block = np.concatenate((concate_direction_block, msdirection_block), axis=0)
        concate_qt_block = concate_qt_block[1:]
        concate_bt_block = concate_bt_block[1:]
        concate_direction_block = concate_direction_block[1:]
        save_qt_path = os.path.join(save_dir, 'Validate_' + comp + '_QP' + str(qp) + '_QTdepth_Block8' + '.npy')
        save_bt_path = os.path.join(save_dir, 'Validate_' + comp + '_QP' + str(qp) + '_BTdepth_Block16' + '.npy')
        save_direction_path = os.path.join(save_dir, 'Validate_' + comp + '_QP' + str(qp) + '_MSdirection_Block16' + '.npy')
        print(save_qt_path)
        print(save_bt_path)
        print(save_direction_path)
        print(concate_qt_block.shape)
        print(concate_bt_block.shape)
        print(concate_direction_block.shape)
        print(concate_qt_block.dtype, concate_bt_block.dtype, concate_direction_block.dtype)
        np.save(save_qt_path, concate_qt_block)
        np.save(save_bt_path, concate_bt_block)
        np.save(save_direction_path, concate_direction_block)

def concate_seqs(comp='Y'):  # comp = 'Y' or 'UV'
    if comp == 'Y':
        block_size = 68
    else:
        block_size = 34
    seqs_info_path = r'.\Cfg\VVC_Test_Sequences.txt'
    root_dir = r'E:\VVC-Fast-Partition-DP\Dataset\Test'
    seqs_info_fp = open(seqs_info_path, 'r')
    data = []
    for line in seqs_info_fp:
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:, 0]
    seqs_path_name = data[:, 1]
    # seqs_width = data[:, 2].astype(np.int64)  # enough bits for calculating h*w
    # seqs_height = data[:, 3].astype(np.int64)
    # seqs_frmnum = data[:, 4].astype(np.int64)
    num_seq = 85
    concate_block = np.zeros((1, block_size, block_size), dtype=np.uint8)
    for i_seq in range(22):
        file_name = seqs_name[i_seq] + '_' + comp + '_Block' + str(block_size) + '.npy'
        file_path = os.path.join(root_dir + r'\Block_' + comp, file_name)
        block_comp = np.load(file_path)
        # print(file_path)
        # print(block_comp.shape)
        block_num = block_comp.shape[0]
        concate_block = np.concatenate((concate_block, block_comp), axis=0)
    concate_block = concate_block[1:]
    save_path = os.path.join(root_dir, 'Test_' + comp + '_Block' + str(block_size) + '.npy')
    print(save_path)
    print(concate_block.shape)
    print(concate_block.dtype)
    np.save(save_path, concate_block)

def concate_partition():
    comp = 'Luma'
    seqs_info_path = r'.\Cfg\Training_Sequences.txt'
    root_dir = r'E:\VVC-Fast-Partition-DP\Dataset'
    seqs_info_fp = open(seqs_info_path, 'r')
    data = []
    for line in seqs_info_fp:
        data.append(line.rstrip('\n').split(','))
    seqs_info_fp.close()
    data = np.array(data)
    print(data.shape)
    seqs_name = data[:, 0]
    seqs_path_name = data[:, 1]
    seqs_width = data[:, 2].astype(np.int64)  # enough bits for calculating h*w
    seqs_height = data[:, 3].astype(np.int64)
    seqs_frmnum = data[:, 4].astype(np.int64)
    for qp in [22, 27, 32, 37]:
        concate_qt_block = np.zeros((1, 8, 8), dtype=np.uint8)
        concate_bt_block = np.zeros((1, 16, 16), dtype=np.uint8)
        concate_direction_block = np.zeros((1, 3, 16, 16), dtype=np.int8)
        for i_seq in range(82, 85):  # need to change
            qt_name = seqs_name[i_seq] + '_QTdepth_QP' + str(qp) + '_' + comp + '.npy'
            bt_name = seqs_name[i_seq] + '_BTdepth_QP' + str(qp) + '_' + comp + '.npy'
            direction_name = seqs_name[i_seq] + '_MSDirection_QP' + str(qp) + '_' + comp + '.npy'
            qt_path = os.path.join(root_dir + r'\Partition_' + comp, qt_name)
            bt_path = os.path.join(root_dir + r'\Partition_' + comp, bt_name)
            direction_path = os.path.join(root_dir + r'\Partition_' + comp, direction_name)
            print(direction_path)
            qt_block = np.load(qt_path)
            bt_block = np.load(bt_path)
            msdirection_block = np.load(direction_path)
            print(qt_block.dtype, bt_block.dtype, msdirection_block.dtype)
            print(seqs_name[i_seq], msdirection_block.shape)
            block_num = msdirection_block.shape[0]
            if block_num < 5000:
                sub_num = 500
            else:
                sub_num = 5000
            concate_qt_block = np.concatenate((concate_qt_block, qt_block[0:sub_num]), axis=0)
            concate_bt_block = np.concatenate((concate_bt_block, bt_block[0:sub_num]), axis=0)
            concate_direction_block = np.concatenate((concate_direction_block, msdirection_block), axis=0)
        concate_qt_block = concate_qt_block[1:]
        concate_bt_block = concate_bt_block[1:]
        concate_direction_block = concate_direction_block[1:]
        save_qt_path = os.path.join(root_dir, 'TestSub_' + comp + '_QP' + str(qp) + '_QTdepth_Block8' + '.npy')
        save_bt_path = os.path.join(root_dir, 'TestSub_' + comp + '_QP' + str(qp) + '_BTdepth_Block16' + '.npy')
        save_direction_path = os.path.join(root_dir + r'\MSDirection_Map', 'Validation_' + comp + '_QP' + str(qp) + '_MSDirection_Block16' + '.npy')
        print(save_direction_path)
        print(concate_qt_block.shape)
        print(concate_bt_block.shape)
        print(concate_direction_block.shape)
        print(concate_qt_block.dtype, concate_bt_block.dtype, concate_direction_block.dtype)
        np.save(save_qt_path, concate_qt_block)
        np.save(save_bt_path, concate_bt_block)
        np.save(save_direction_path, concate_direction_block)

def generate_cfg():
    root_dir = r'G:\Research\testyuv\Dataset\Train'
    cfg_dir = r'.\Cfg'
    seqs_info_path = r'.\Cfg\Training_Sequences.txt'
    seqs_info_fp = open(seqs_info_path, 'w')
    for count, item in enumerate(os.listdir(root_dir)):
        if count >= 500:
            break
        if 'yuv' in item:
            seq_name = item.rstrip('.yuv')
            seq_size = item.rstrip('.yuv').split('_')[-3]
            seq_width = seq_size.split('x')[0]
            seq_height = seq_size.split('x')[1]
            seq_frm_rate = item.rstrip('.yuv').split('_')[-2]
            seq_frm_num = item.rstrip('.yuv').split('_')[-1]
            # print(seq_name)
            seq_cfg_path = os.path.join(cfg_dir, seq_name + '.cfg')
            seq_cfg_fp = open(seq_cfg_path, 'w')
            seq_cfg_fp.write('#======== File I/O =========\n')
            seq_cfg_fp.write(
                'InputFile            : //C01/share/data/origCfp/SequenceData/Train/' + seq_name + '.yuv\n')
            seq_cfg_fp.write('InputBitDepth        : 8           # Input bitdepth\n')
            seq_cfg_fp.write('FrameRate            : ' + seq_frm_rate + '          # Frame per second\n')
            seq_cfg_fp.write('FrameSkip            : 0           # Number of frames to be skipped in input\n')
            seq_cfg_fp.write('SourceWidth          : ' + seq_width + '        # Input  frame width\n')
            seq_cfg_fp.write('SourceHeight         : ' + seq_height + '        # Input  frame height\n')
            seq_cfg_fp.write('FramesToBeEncoded    : ' + seq_frm_num + '         # Number of frames to be coded\n')
            seq_cfg_fp.write('\n')
            seq_cfg_fp.write('Level                : 4.1\n')
            seq_cfg_fp.close()
            seqs_info_fp.write(
                seq_name + ',' + seq_width + ',' + seq_height + ',' + seq_frm_num + ',' + seq_frm_rate + '\n')
            # check data
            file_path = os.path.join(root_dir, item)
            file_size = os.path.getsize(file_path)
            cal_file_size = int(seq_width) * int(seq_height) * int(seq_frm_num) * 1.5
            if file_size == cal_file_size:
                print('True')
            else:
                print('*******************False')
    seqs_info_fp.close()

def load_sequences_info():
    num = 82
    seqs_info_path = r"E:\VVC-Fast-Partition-DP\Code\Cfg\Training_Sequences.txt"
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
        # SubSampleRatio = 8
        sub_frmnum = (seqs_frmnum[i] + SubSampleRatio - 1) // SubSampleRatio
        sub_frmnum_list.append(sub_frmnum)
    return seqs_path_name, seqs_width, seqs_height, sub_frmnum_list


if __name__ == '__main__':
    print("Start...")
    # save_sequence_block_set()
    # save_partition_block_set()
    # concate_seqs()
    # concate_partition()
    print("End!!!")




