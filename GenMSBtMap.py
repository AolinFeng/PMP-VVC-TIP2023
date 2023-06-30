'''
Function:
  Form [incomplete partition map] to [partition map] (mainly generate the multi-layer bt depth map)

Main functions:
  * main_process()

Note:
  * The basic idea of this code is using the way of post-processing to get partition map.
    So you must understand the post-processing algorithm to understand this code
    This code file is very similar to the Map2Partition.py that is used for post-processing.
  * When it comes to the QTBT structure rather than QTMTT, the way to get partition map can be much easier.
    Dealing with the ternary partition mode is a headache.

Author: Aolin Feng
'''

import os
import argparse
import numpy as np
# import torch
from matplotlib import pyplot as plt
# from Metrics import eli_structual_error
# import time
# qt depth 0->3
# bt depth 0->5 0->4 (chroma)
# direction 0 1 2

def delete_tree(root):
    if len(root.children) == 0:  # no child
        del root
    else:
        children = root.children
        for child in children:
            delete_tree(child)

# Split_Mode and Search are set for generate combination modes
# example: input [[1,2], [3,4,5]]; output [[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]]

class Split_Node():
    def __init__(self, split_type):
        self.split_type = split_type
        self.children = []

class Search():
    def __init__(self, cus_candidate_mode_list):
        self.split_root = Split_Node(0)
        self.parent_list = []
        self.cus_mode_list = cus_candidate_mode_list
        self.partition_modes = []
        self.cus_modes = []

    def get_cus_mode_tree(self):
        self.parent_list = [self.split_root]
        while len(self.cus_mode_list) != 0:
            parent_temp = []
            for parent in self.parent_list:
                for split_type in self.cus_mode_list[0]:
                    child = Split_Node(split_type)
                    parent.children.append(child)
                    parent_temp.append(child)
            self.parent_list = parent_temp
            self.cus_mode_list.pop(0)

    def bfs(self, node):
        self.cus_modes.append(node.split_type)
        if len(node.children) == 0:
            temp = self.cus_modes[1:]
            self.partition_modes.append(temp)
            self.cus_modes.pop(-1)
        else:
            for child in node.children:
                self.bfs(child)
            self.cus_modes.pop(-1)

    def get_partition_modes(self):
        self.get_cus_mode_tree()
        self.bfs(self.split_root)
        return self.partition_modes

class Map_Node():
    def __init__(self, bt_map, mtt_depth, cus, parent=None):
        self.bt_map = bt_map
        self.mtt_depth = mtt_depth
        self.cus = cus  # [x, y, h, w] list
        self.children = []
        self.parent = parent

class Map_to_SubMap():
    """Convert Partition maps to Split flags to Partition vectors"""
    def __init__(self, qt_map, bt_map, dire_map, chroma_factor, lamb1=0.8, lamb2=1.0, lamb3=1.2, lamb4=0.2, lamb5=0.2):
        self.qt_map = qt_map
        self.bt_map = bt_map
        self.dire_map = dire_map
        self.chroma_factor = chroma_factor
        self.par_vec = np.zeros((2, 17, 17), dtype=np.uint8)
        self.sub_map = np.zeros((3, 16, 16), dtype=np.uint8)
        self.cur_leaf_nodes = []  # store leaf nodes of Map Tree
        # lamb indicates several kinds of thresholds
        self.lamb1 = lamb1  # control no partition based on depth map
        self.lamb2 = lamb2  # judge direction based on direction map
        self.lamb3 = lamb3  # control hor or ver
        self.lamb4 = lamb4  # control number of minus
        self.lamb5 = lamb5  # control number of zero
        self.time = 0

    def split_cur_map(self, x, y, h, w, split_type):
        # split current cu [x,y,h,w]
        # split_type: 0 1 2 3 4 no bth btv tth ttv
        if split_type == 0:
            return [[x, y, h, w]]
        elif split_type == 1:  # bth
            return [[x, y, h//2, w], [x+h//2, y, h//2, w]]
        elif split_type == 2:  # btv
            return [[x, y, h, w//2], [x, y+w//2, h, w//2]]
        elif split_type == 3:  # tth
            return [[x, y, h//4, w], [x+h//4, y, h//2, w], [x+(h*3)//4, y, h//4, w]]
        elif split_type == 4:  # ttv
            return [[x, y, h, w//4], [x, y+w//4, h, w//2], [x, y+(w*3)//4, h, w//4]]
        else:
            print("Unknown split type!")

    def can_split_mode_list(self, x, y, h, w, cur_bt_map, mtt_depth):
        """output candidate split type list for current cu"""
        comp_map = self.bt_map[x:x+h, y:y+w] - cur_bt_map[x:x+h, y:y+w]
        count_zero = len(np.where(comp_map == 0)[0])  # number of zero
        if count_zero >= self.lamb1 * h * w:  # no partition
            return [0]
        count_hor = len(np.where(self.dire_map[mtt_depth, x:x+h, y:y+w] == 1)[0])  # horizontal unit number of current direction map
        count_ver = len(np.where(self.dire_map[mtt_depth, x:x+h, y:y+w] == -1)[0])  # vertical unit number of current direction map

        direction = 0  # 0 1 2 Unknown Horizontal Vertical
        if (count_ver + count_hor) >= self.lamb2 * h * w:  # non-flat flag within direction map dominates
            if count_hor >= self.lamb3 * count_ver:
                direction = 1
            elif count_ver >= self.lamb3 * count_hor:
                direction = 2
        else:
            return [0]

        initial_split_list = []
        for split_mode in [1, 2, 3, 4]:
            if split_mode == 1 and (h // (2*self.chroma_factor) == 0 or h % (2*self.chroma_factor) != 0):  # bth
                continue
            if split_mode == 2 and (w // (2*self.chroma_factor) == 0 or w % (2*self.chroma_factor) != 0):  # btv
                continue
            if split_mode == 3 and (h // (4*self.chroma_factor) == 0 or h % (4*self.chroma_factor) != 0):  # tth
                continue
            if split_mode == 4 and (w // (4*self.chroma_factor) == 0 or w % (4*self.chroma_factor) != 0):  # ttv
                continue
            if (split_mode == 1 or split_mode == 3) and direction == 2:  # horizontal mode with vertical texture
                continue
            if (split_mode == 2 or split_mode == 4) and direction == 1:  # vertical mode with horizontal texture
                continue
            initial_split_list.append(split_mode)

        candidate_mode_list = []
        bt_map_temp = np.zeros_like(cur_bt_map, dtype=np.int8)
        for split_mode in initial_split_list:  # try potential partitions
            sub_map_xyhw = self.split_cur_map(x, y, h, w, split_mode)
            bt_map_temp[:, :] = cur_bt_map[:, :]
            split_thres = 0
            for sub_map_id in range(len(sub_map_xyhw)):  # traverse all sub-blocks
                [sub_x, sub_y, sub_h, sub_w] = sub_map_xyhw[sub_map_id]
                # comp_map defines the difference between the bt map and the temp bt map;
                # the proper partition would bring a small portion of negative values in the map;
                # if zero appears in the map, a proper partition would expect the portion of zero values either very large (partition end) or very small
                bt_map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                if (split_mode == 3 or split_mode == 4) and (sub_map_id != 1):
                    # depth +2 in the first and last parts fot tt partition
                    bt_map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                comp_map = self.bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] - bt_map_temp[
                                                                                   sub_x:sub_x + sub_h,
                                                                                   sub_y:sub_y + sub_w]

                count_minus = len(np.where(comp_map < 0)[0])  # number of minus
                count_zero = len(np.where(comp_map == 0)[0])  # number of zero
                num_pixel = sub_h * sub_w
                if count_minus < num_pixel * self.lamb4 and (
                        count_zero < num_pixel * self.lamb5 or count_zero > num_pixel * (1 - self.lamb5)):
                    # current sub-block gets proper partition
                    split_thres += 1
            if split_thres == len(sub_map_xyhw):  # all sub-block get proper partition
                candidate_mode_list.append(split_mode)

        return candidate_mode_list

    def get_candidate_map_tree(self, map_node):
        if map_node.mtt_depth >= 3:
            return
        cur_cus = map_node.cus
        cu_num = len(cur_cus)
        cur_bt_map = map_node.bt_map
        cur_mtt_depth = map_node.mtt_depth
        cus_candidate_mode_list = []
        for i in range(cu_num):
            cus_candidate_mode_list.append([])  # store all candidate split modes of every CU
        for cu_id in range(cu_num):  # traverse all CUs in current map
            [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]
            candidate_mode_list = self.can_split_mode_list(cu_x, cu_y, cu_h, cu_w, cur_bt_map, cur_mtt_depth)
            if len(candidate_mode_list) == 0:  # no proper partition for a certain CU
                return
            cus_candidate_mode_list[cu_id] += candidate_mode_list
        # t1 = time.time()
        # partition_modes = []  # store all possible combination modes of all CUs
        # for comb_mode_id in range(1, 7**cu_num):
        #     # maximum possible CU modes combination 5**cu_num, 0 means all CUs no partition
        #     cus_modes = []
        #     for cu_id in range(cu_num):
        #         bit_number = 7 << (cu_id * 3)
        #         mode_id = (comb_mode_id & bit_number) >> (cu_id * 3)
        #         if mode_id not in cus_candidate_mode_list[cu_id]:
        #             break
        #         cus_modes.append(mode_id)
        #     if len(cus_modes) == cu_num:
        #         partition_modes.append(cus_modes)
        # the function of above annotation codes is equivalent to the following two lines of codes, but inefficient
        s = Search(cus_candidate_mode_list)
        partition_modes = s.get_partition_modes()

        # self.time += time.time() - t1
        for cus_modes in partition_modes:  # traverse all possible combination cu split modes
            child_bt_map = np.zeros_like(cur_bt_map, dtype=np.uint8)
            child_bt_map[:, :] = cur_bt_map
            child_cus = []
            for cu_id in range(cu_num):  # traverse all cu
                [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]  # location and size of current CU
                cu_mode = cus_modes[cu_id]  # split mode of current CU
                child_map_xyhw = self.split_cur_map(cu_x, cu_y, cu_h, cu_w, cu_mode)
                child_cus += child_map_xyhw
                if cu_mode == 0:  # no partition
                    continue
                for sub_block_id in range(len(child_map_xyhw)):  # traverse all sub-blocks
                    [sub_x, sub_y, sub_h, sub_w] = child_map_xyhw[sub_block_id]
                    child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                    if (cu_mode == 3 or cu_mode == 4) and (sub_block_id != 1):
                        # depth +2 in the first and last parts
                        child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
            child_map_node = Map_Node(bt_map=child_bt_map, mtt_depth=cur_mtt_depth+1, cus=child_cus, parent=map_node)
            self.get_candidate_map_tree(child_map_node)
            map_node.children.append(child_map_node)

    def get_leaf_nodes(self, map_node):
        if len(map_node.children) == 0:  # no children node
            self.cur_leaf_nodes.append(map_node)
        else:
            for child_node in map_node.children:
                self.get_leaf_nodes(child_node)

    def print_tree(self, map_node, depth):
        print('**********************')
        print('node', depth)
        print(map_node.mtt_depth)
        print(map_node.bt_map)
        print(map_node.cus)
        print(len(map_node.children))
        print('**********************')
        if len(map_node.children) != 0:
            for child_node in map_node.children:
                self.print_tree(child_node, depth+1)

    def set_bt_partition_vector(self, x, y, h, w):
        init_bt_map = np.zeros((16, 16), dtype=np.int8)
        map_root = Map_Node(bt_map=init_bt_map, mtt_depth=0, cus=[[x, y, h, w]])
        self.get_candidate_map_tree(map_root)  # build Map Tree

        # self.print_tree(map_root, 0)
        self.cur_leaf_nodes = []
        self.get_leaf_nodes(map_root)  # get lead nodes list of Map Tree

        error_list = []
        for leaf_node in self.cur_leaf_nodes:
            leaf_bt_map = leaf_node.bt_map
            error = np.sum(np.abs(leaf_bt_map[x:x+h, y:y+w] - self.bt_map[x:x+h, y:y+w]))
            error_list.append(error)
        min_index = error_list.index(min(error_list))
        best_bt_map = self.cur_leaf_nodes[min_index].bt_map  # best bt map
        best_cus = self.cur_leaf_nodes[min_index].cus  # best partition
        # print('*********************************************************')
        # print('error_list', error_list, np.min(error_list))
        # print('x, y, h, w', x, y, h, w)
        # print(best_bt_map[x:x + h, y:y + w])
        # print(self.cur_leaf_nodes[2].cus)
        delete_tree(map_root)
        for cu in best_cus:
            [cu_x, cu_y, cu_h, cu_w] = cu
            for i_w in range(cu_w):  # set CU horizontal edges
                self.par_vec[0, cu_x, cu_y + i_w] = 1
                self.par_vec[0, cu_x + cu_h, cu_y + i_w] = 1
            for i_h in range(cu_h):  # set CU vertical edges
                self.par_vec[1, cu_x + i_h, cu_y] = 1
                self.par_vec[1, cu_x + i_h, cu_y + cu_w] = 1

    def set_partition_vector(self, depth, qx, qy):
        cur_qt_depth = self.qt_map[qx, qy]
        sub_map_size = 8 >> depth
        if cur_qt_depth == depth:  # end QT partition [2*qx:2*qx + 2*sub_map_size, 2*qy:2*qy + 2*sub_map_size]
            self.set_bt_partition_vector(2*qx, 2*qy, 2*sub_map_size, 2*sub_map_size)
            return
        elif cur_qt_depth > depth:  # carry on QT partition
            for i in range(sub_map_size * 2):
                # set qt node partition
                self.par_vec[0, 2 * qx + sub_map_size, 2 * qy + i] = 1  # horizontal
                self.par_vec[1, 2 * qx + i, 2 * qy + sub_map_size] = 1  # vertical
            for i_offset in range(2):
                for j_offset in range(2):
                    self.set_partition_vector(depth + 1, qx + i_offset * sub_map_size // 2, qy + j_offset * sub_map_size // 2)

            return
    def get_partition(self):
        self.set_partition_vector(0, 0, 0)
        return self.par_vec

    def set_sub_map(self, depth, qx, qy):
        cur_qt_depth = self.qt_map[qx, qy]
        sub_map_size = 8 >> depth
        if cur_qt_depth == depth:  # end QT partition [2*qx:2*qx + 2*sub_map_size, 2*qy:2*qy + 2*sub_map_size]
            self.set_bt_sub_map(2*qx, 2*qy, 2*sub_map_size, 2*sub_map_size)
            return
        elif cur_qt_depth > depth:  # carry on QT partition
            for i_offset in range(2):
                for j_offset in range(2):
                    self.set_sub_map(depth + 1, qx + i_offset * sub_map_size // 2, qy + j_offset * sub_map_size // 2)
            return

    def set_bt_sub_map(self, x, y, h, w):
        init_bt_map = np.zeros((16, 16), dtype=np.int8)
        map_root = Map_Node(bt_map=init_bt_map, mtt_depth=0, cus=[[x, y, h, w]])
        self.get_candidate_map_tree(map_root)  # build Map Tree

        # self.print_tree(map_root, 0)
        self.cur_leaf_nodes = []
        self.get_leaf_nodes(map_root)  # get lead nodes list of Map Tree

        error_list = []
        for leaf_node in self.cur_leaf_nodes:
            leaf_bt_map = leaf_node.bt_map
            error = np.sum(np.abs(leaf_bt_map[x:x+h, y:y+w] - self.bt_map[x:x+h, y:y+w]))
            error_list.append(error)
        min_index = error_list.index(min(error_list))
        best_node = self.cur_leaf_nodes[min_index]
        best_node2 = best_node.parent
        best_node1 = best_node2.parent
        best_node0 = best_node1.parent
        best_cus = best_node.cus  # best partition
        best_bt_map = best_node.bt_map  # best bt map
        best_bt_map2 = best_node2.bt_map
        best_bt_map1 = best_node1.bt_map
        best_bt_map0 = best_node0.bt_map
        # print(np.sum(self.bt_map[x:x+h, y:y+w]-best_bt_map[x:x+h, y:y+w]))
        # print(best_node.mtt_depth)
        # print(best_bt_map0)
        # print(best_bt_map1)
        # print(best_bt_map2)
        # print(best_bt_map)
        # print('*********************************************************')
        # print('error_list', error_list, np.min(error_list))
        # print('x, y, h, w', x, y, h, w)
        # print(best_bt_map[x:x + h, y:y + w])
        # print(self.cur_leaf_nodes[2].cus)
        self.sub_map[0, x:x+h, y:y+w] = best_bt_map1[x:x+h, y:y+w]
        self.sub_map[1, x:x+h, y:y+w] = best_bt_map2[x:x+h, y:y+w]
        self.sub_map[2, x:x+h, y:y+w] = best_bt_map[x:x+h, y:y+w]
        delete_tree(map_root)

    def get_sub_map(self):
        self.set_sub_map(0, 0, 0)
        return self.sub_map

def getSubMap(qt_map, bt_map, dire_map, chroma_factor):
    # start_time = time.time()
    partition = Map_to_SubMap(qt_map, bt_map, dire_map, chroma_factor)
    out_sub_map = partition.get_sub_map()
    # total_time = time.time() - start_time
    return out_sub_map

def map_to_parititon(qt_map, bt_map, dire_map, chroma_factor):
    # start_time = time.time()
    partition = Map_to_SubMap(qt_map, bt_map, dire_map, chroma_factor)
    p = partition.get_partition()
    # total_time = time.time() - start_time
    return p[0][:16, :16], p[1][:16, :16]

def get_sequence_partition_for_VTM(qt_map, bt_map, dire_map, is_luma, save_path, frm_num, frm_width, frm_height):
    # partition maps --> partition edge vector + sequence qt depth map + sequence direction map
    # dire_map = np.where(dire_map < 0, np.ones_like(dire_map) * 2, dire_map)
    chroma_factor = 2
    if is_luma:
        chroma_factor = 1
    if save_path is not None:
        out_file = open(save_path, 'w')
    block_num_in_height = frm_height // 64
    block_num_in_width = frm_width // 64
    seq_partition_hor_mat = np.zeros((frm_num, block_num_in_height * 16, block_num_in_width * 16))
    seq_partition_ver_mat = np.zeros((frm_num, block_num_in_height * 16, block_num_in_width * 16))
    seq_qt_map = np.zeros((frm_num, block_num_in_height * 8, block_num_in_width * 8))
    seq_dire_map = np.zeros((frm_num, 3, block_num_in_height * 16, block_num_in_width * 16))
    for frm_id in range(frm_num):
        print("Frame ", frm_id)
        frm_block_id = frm_id * block_num_in_height * block_num_in_width
        for block_x in range(block_num_in_height):
            for block_y in range(block_num_in_width):
                block_id = frm_block_id + block_x * block_num_in_width + block_y
                hor_mat, ver_mat = map_to_parititon(qt_map[block_id], bt_map[block_id], dire_map[block_id], chroma_factor)
                seq_partition_hor_mat[frm_id, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = hor_mat
                seq_partition_ver_mat[frm_id, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = ver_mat
                seq_qt_map[frm_id, block_x * 8:(block_x + 1) * 8, block_y * 8:(block_y + 1) * 8] = qt_map[block_id]
                seq_dire_map[frm_id, :, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = dire_map[block_id]
        if save_path is not None:
            hor_vec = seq_partition_hor_mat[frm_id].reshape(-1).astype(np.uint8)
            ver_vec = seq_partition_ver_mat[frm_id].reshape(-1).astype(np.uint8)
            qtdepth_vec = seq_qt_map[frm_id].reshape(-1).astype(np.uint8)
            dire_vec = seq_dire_map[frm_id].reshape(-1).astype(np.uint8)
            for i in range(hor_vec.size):  # horizontal edge vector
                out_file.write(str(hor_vec[i]) + '\n')
            for i in range(ver_vec.size):  # vertical edge vector
                out_file.write(str(ver_vec[i]) + '\n')
            for i in range(qtdepth_vec.size):  # qt depth vector
                out_file.write(str(qtdepth_vec[i]) + '\n')
            for i in range(dire_vec.size):  # direction vector
                out_file.write(str(dire_vec[i]) + '\n')
            # print(hor_vec.size)
            # print(qtdepth_vec.size)
            # print(dire_vec.size)
    if save_path is not None:
        out_file.close()
    check_frm_id = 2
    partition = np.clip(seq_partition_hor_mat[check_frm_id] + seq_partition_ver_mat[check_frm_id], a_min=0, a_max=1)
    plt.imshow(partition, cmap='gray')
    plt.axis("off")
    plt.savefig("Partition.png", dpi=640, bbox_inches='tight', pad_inches=0)
    plt.show()

def gen_seq_sub_map(qt_map, bt_map, dire_map, is_luma):
    # partition maps --> partition edge vector + sequence qt depth map + sequence direction map
    # dire_map = np.where(dire_map < 0, np.ones_like(dire_map) * 2, dire_map)
    chroma_factor = 2
    if is_luma:
        chroma_factor = 1
    block_num = bt_map.shape[0]
    out_msbt_block = np.zeros((block_num, 3, 16, 16), dtype=np.uint8)
    for block_id in range(block_num):
        if block_id % 10000 == 0:
            print(block_id)
        out_sub_map = getSubMap(qt_map[block_id], bt_map[block_id], dire_map[block_id], chroma_factor)
        # print(out_sub_map)
        out_msbt_block[block_id] = out_sub_map

    return out_msbt_block

def main_process(args):
    data_type_list = ["Train"]
    qp_base_list = [22, 27, 32, 37]
    qp_list = []
    for qp_id in range(args.startQPID, args.startQPID + args.qpNum):
        qp_list.append(qp_base_list[qp_id])
    print("QP:", qp_list)

    for comp in ["Luma", "Chroma"]:
        pre_path = r"D:\fengal\VVCFast-Partition-DP\Dataset\Label"
        pre_save_path = r"D:\fengal\VVCFast-Partition-DP\Dataset\Label\MSBT"
        is_luma = True
        if comp == "Chroma":
            is_luma = False
        for qp in qp_list:
            for data_type in data_type_list:
                qt_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "QTdepth_Block8.npy"
                bt_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "BTdepth_Block16.npy"
                dire_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "MSdirection_Block16.npy"
                qt_path = os.path.join(pre_path, qt_name)
                bt_path = os.path.join(pre_path, bt_name)
                dire_path = os.path.join(pre_path, dire_name)
                print(qt_path)
                print(bt_path)
                print(dire_path)

                qt_map = np.load(qt_path) - 1
                bt_map = np.load(bt_path)
                dire_map = np.load(dire_path)
                # print(np.max(bt_map), np.min(bt_map))
                # print(dire_map.shape)

                out_msbt_block = gen_seq_sub_map(qt_map=qt_map, bt_map=bt_map, dire_map=dire_map, is_luma=True)
                print(out_msbt_block.shape)
                save_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "MSBTdepth_Block16.npy"
                save_path = os.path.join(pre_save_path, save_name)
                np.save(save_path, out_msbt_block)
                print("*******************Validate**********************")
                print(np.sum(bt_map - out_msbt_block[:, 2, :, :]))

def test():
    data_type_list = ["RaceHorseC128"]
    qp_list = [32]
    for comp in ["Luma"]:
        pre_path = r".\Temp"
        pre_save_path = r".\Temp"
        is_luma = True
        if comp == "Chroma":
            is_luma = False
        for qp in qp_list:
            for data_type in data_type_list:
                qt_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "QTdepth_Block8.npy"
                bt_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "BTdepth_Block16.npy"
                dire_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "MSdirection_Block16.npy"
                qt_path = os.path.join(pre_path, qt_name)
                bt_path = os.path.join(pre_path, bt_name)
                dire_path = os.path.join(pre_path, dire_name)
                print(qt_path)
                print(bt_path)
                print(dire_path)

                qt_map = np.load(qt_path) - 1
                bt_map = np.load(bt_path)
                dire_map = np.load(dire_path)
                # print(np.max(bt_map), np.min(bt_map))
                # print(dire_map.shape)

                out_msbt_block = gen_seq_sub_map(qt_map=qt_map, bt_map=bt_map, dire_map=dire_map, is_luma=True)
                print(out_msbt_block.shape)
                save_name = data_type + '_' + comp + "_QP" + str(qp) + '_' + "MSBTdepth_Block16.npy"
                save_path = os.path.join(pre_save_path, save_name)
                np.save(save_path, out_msbt_block)
                print("*******************Validate**********************")
                print(np.sum(bt_map - out_msbt_block[:, 2, :, :]))

if __name__ == '__main__':
    print('start running ...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--startQPID', default=0, type=int, help='QP start ID')
    parser.add_argument('--qpNum', default=1, type=int, help='test QP number')
    args = parser.parse_args()
    # test()
    main_process(args)
    print("End !!!")


