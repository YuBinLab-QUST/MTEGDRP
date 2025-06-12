# -*- coding: utf-8 -*-
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch._C import device
from Model_utils_1204 import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True


"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "MTEGDRP-main/data/"


def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "r")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs


def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'w')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'w')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)


def cid_from_other_source():
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict:
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k: v for k, v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict


def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k, v in cids_dict.iteritems()]
    inv_cids_dict = {v: k for k, v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES',
             overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES:{smile}")

    # 加氢
    mol = Chem.AddHs(mol)

    # 尝试生成多个三维构象
    num_confs = 10
    ids = AllChem.EmbedMultipleConfs(mol,numConfs=num_confs)

    # 检查是否成功嵌入
    if len(ids) == 0:
        raise ValueError(f"Embedding failed for SMILES:{smile}")

    # 优化第一个有效构象
    if AllChem.UFFOptimizeMolecule(mol,confId=ids[0]) == -1:
        raise ValueError(f"UFF optimization failed for SMILES:{smile}")

    c_size = mol.GetNumAtoms()

    # 原子特征
    features = []
    coordinates = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

        # 获取原子坐标并转换为numpy数组
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        coordinates.append(np.array(pos))

    # 边
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # 创建图并获取边索引
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # 创建邻接矩阵
    adjacency_matrix = np.zeros((c_size, c_size), dtype=np.float32)
    for e1, e2 in edges:
        adjacency_matrix[e1, e2] = 1
        adjacency_matrix[e2, e1] = 1  # 无向图，矩阵对称
    # 设置对角线为1，表示每个节点与自己有连接
    np.fill_diagonal(adjacency_matrix, 1)

    # 创建距离矩阵
    distance_matrix = np.zeros((c_size, c_size), dtype=np.float32)
    coordinates = np.array(coordinates)
    for i in range(c_size):
        for j in range(c_size):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    return c_size, features, edge_index,coordinates,adjacency_matrix,distance_matrix


def load_drug_smile():
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]  # drug_name:(SNX-2112)
        smile = item[2]

        if name != "Bleomycin":
            if name in drug_dict:
                pos = drug_dict[name]
            else:
                pos = len(drug_dict)
                drug_dict[name] = pos
            drug_smile.append(smile)

    smile_graph = {}
    for smile in drug_smile:
        try:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        except ValueError as e:
            print(e)
            continue
    return drug_dict, drug_smile, smile_graph


def save_cell_mut_matrix():
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    # features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)

    cell_dict_mut = cell_dict
    cell_feature_mut = cell_feature
    return cell_dict_mut, cell_feature_mut, mut_dict


def save_cell_meth_matrix():
    f = open(folder + "METH_CELLLINES_BEMs_PANCAN.csv")
    reader = csv.reader(f)
    next(reader)
    # numberCol = len(next(reader)) - 1
    # features = {}
    cell_dict = {}
    matrix_list = []
    meth_dict = {}
    for item in reader:
        cell_id = item[1]
        meth = item[2]
        is_mutated = int(item[3])

        if meth in meth_dict:
            col = meth_dict[meth]
        else:
            col = len(meth_dict)
            meth_dict[meth] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(meth_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('meth_dict', 'wb') as fp:
        pickle.dump(meth_dict, fp)

    cell_dict_meth = cell_dict
    cell_feature_meth = cell_feature
    return cell_dict_meth, cell_feature_meth, meth_dict


def save_cell_ge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.csv")
    reader = csv.reader(f)
    # firstRow = next(reader)
    # numberCol = len(firstRow) - 1
    # features = {}
    cell_dict = {}
    # matrix_list = []
    for item in reader:
        cell_id = item[0]
        ge = []
        for i in range(1, len(item)):
            ge.append(int(item[i]))
        cell_dict[cell_id] = np.asarray(ge)
    return cell_dict


def save_cell_oge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.txt")
    line = f.readline()
    elements = line.split()
    cell_names = []
    feature_names = []
    cell_dict = {}
    i = 0
    for cell in range(2, len(elements)):
        if i < 500:
            cell_name = elements[cell].replace("DATA.", "")
            cell_names.append(cell_name)
            cell_dict[cell_name] = []

    min = 0
    max = 12
    for line in f.readlines():
        elements = line.split("\t")
        if len(elements) < 2:
            print(line)
            continue
        feature_names.append(elements[1])

        for i in range(2, len(elements)):
            cell_name = cell_names[i - 2]
            value = float(elements[i])
            if min == 0:
                min = value
            if value < min:
                min = value
            if max < value:
                max = value
            cell_dict[cell_name].append(value)

    cell_feature = []
    for cell_name in cell_names:
        for i in range(0, len(cell_dict[cell_name])):
            cell_dict[cell_name][i] = (cell_dict[cell_name][i] - min) / (max - min)
        cell_dict[cell_name] = np.asarray(cell_dict[cell_name])
        cell_feature.append(np.asarray(cell_dict[cell_name]))

    cell_feature = np.asarray(cell_feature)

    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1
    cell_dict_ge = cell_dict
    cell_feature_ge = cell_feature
    ge_list = cell_names
    return cell_dict_ge, cell_feature_ge, ge_list


"""
假设在处理之前 cell_dict 的内容如下：

cell_dict = {'cell1': [0.1, 0.2, 0.3], 'cell2': [0.4, 0.5, 0.6]}
经过这段代码处理后：

i 从 0 开始递增，cell_dict 中的每个细胞名称将被替换为相应的整数索引：

cell_dict = {'cell1': 0, 'cell2': 1}
cell_dict_ge 和 cell_feature_ge 将分别保存更新后的 cell_dict 和特征数据：

cell_dict_ge = {'cell1': 0, 'cell2': 1}
cell_feature_ge = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
ge_list 将包含所有细胞的名称：

ge_list = ['cell1', 'cell2']
"""


class DataBuilder(Dataset):
    def __init__(self, cell_feature_ge):
        self.cell_feature_ge = cell_feature_ge
        self.cell_feature_ge = torch.FloatTensor(self.cell_feature_ge)
        self.len = self.cell_feature_ge[0]

    def __getitem__(self, index):
        return self.cell_feature_ge[index]

    def __len__(self):
        return self.len


def save_mix_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict_mut, cell_feature_mut, mut_dict = save_cell_mut_matrix()
    cell_dict_meth, cell_feature_meth, meth_dict = save_cell_meth_matrix()
    cell_dict_ge, cell_feature_ge, ge_list = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()
    print(f"drug_dict 的长度为：{len(drug_dict)}")
    print(f"drug_smile 的长度为：{len(drug_smile)}")
    print(f"smile_graph 的长度为：{len(smile_graph)}")

    temp_data = []

    bExist = np.zeros((len(drug_dict), len(cell_dict_mut)))

    for item in reader:
        drug = item[0]  # drug_name:(Erlotinib)
        cell = item[3]  # Cosmic sample id:(683665)
        ic50 = item[8]  # ic50值:(2.43658649)
        ic50 = np.array(ic50).astype(float)
        # ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc_mut = []
    xc_meth = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []

    mut_id = []
    mut_cell_id = []
    meth_id = []
    meth_cell_id = []
    ge_id = []
    ge_cell_id = []

    random.shuffle(temp_data)

    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_ge = kpca.fit_transform(cell_feature_ge)
    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_mut = kpca.fit_transform(cell_feature_mut)
    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_meth = kpca.fit_transform(cell_feature_meth)

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth and cell in cell_dict_mut:
            # if drug in drug_dict and cell in cell_dict_ge:
            # if drug in drug_dict and cell in cell_dict_meth:
            # if drug in drug_dict and cell in cell_dict_mut:

            xd.append(drug_smile[drug_dict[drug]])
            xc_mut.append(cell_feature_mut[cell_dict_mut[cell]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])
            xc_meth.append(cell_feature_meth[cell_dict_meth[cell]])

            y.append(ic50)

            mut_id.append(list(mut_dict.keys()))
            mut_cell_id.append(list(cell_dict_mut.keys()))
            meth_id.append(list(meth_dict.keys()))
            meth_cell_id.append(list(cell_dict_meth.keys()))
            ge_id.append(ge_list)
            ge_cell_id.append(list(cell_dict_ge.keys()))

            bExist[drug_dict[drug], cell_dict_mut[cell]] = 1

            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd = np.asarray(xd)
    xc_mut = np.asarray(xc_mut)
    xc_ge = np.asarray(xc_ge)
    xc_meth = np.asarray(xc_meth)
    y = np.asarray(y)

    mut_id = np.asarray(mut_id)
    mut_cell_id = np.asarray(mut_cell_id)
    meth_id = np.asarray(meth_id)
    meth_cell_id = np.asarray(meth_cell_id)
    ge_id = np.asarray(ge_id)
    ge_cell_id = np.asarray(ge_cell_id)

    size = int(xd.shape[0] * 0.95)
    size1 = int(xd.shape[0] * 0.95)
    size2 = int(xd.shape[0] * 0.9)

    np.save('list_drug_mix_test', lst_drug[size1:])
    np.save('list_cell_mix_test', lst_cell[size1:])
    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)

    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)

    mut_id_test = mut_id[size1:]
    pd.DataFrame(mut_id_test).to_csv("MTEGDRP-main/data/test_data/mut_id_test.csv")
    mut_cell_id_test = mut_cell_id[size1:]
    pd.DataFrame(mut_cell_id_test).to_csv("MTEGDRP-main/data/test_data/mut_cell_id_test.csv")

    meth_id_test = meth_id[size1:]
    pd.DataFrame(meth_id_test).to_csv("MTEGDRP-main/data/test_data/meth_id_test.csv")
    meth_cell_id_test = meth_cell_id[size1:]
    pd.DataFrame(meth_cell_id_test).to_csv("MTEGDRP-main/data/test_data/meth_cell_id_test.csv")

    ge_id_test = ge_id[size1:]
    pd.DataFrame(ge_id_test).to_csv("MTEGDRP-main/data/test_data/ge_id_test.csv")
    ge_cell_id_test = ge_cell_id[size1:]
    pd.DataFrame(ge_cell_id_test).to_csv("MTEGDRP-main/data/test_data/ge_cell_id_test.csv")

    xd_train = xd[:size]
    xd_val = xd[size2:size1]
    xd_test = xd[size1:]
    # pd.DataFrame(xd_train).to_csv("data/test_data/xd_train.csv")
    # pd.DataFrame(xd_val).to_csv("data/test_data/xd_val.csv")
    pd.DataFrame(xd_test).to_csv("MTEGDRP-main/data/test_data/xd_test.csv")

    xc_ge_train = xc_ge[:size]
    xc_ge_val = xc_ge[size2:size1]
    xc_ge_test = xc_ge[size1:]
    # pd.DataFrame(xc_ge_train).to_csv("data/test_data/xc_ge_train.csv")
    # pd.DataFrame(xc_ge_val).to_csv("data/test_data/xc_ge_val.csv")
    pd.DataFrame(xc_ge_test).to_csv("MTEGDRP-main/data/test_data/xc_ge_test.csv")

    xc_meth_train = xc_meth[:size]
    xc_meth_val = xc_meth[size2:size1]
    xc_meth_test = xc_meth[size1:]
    # pd.DataFrame(xc_meth_train).to_csv("data/test_data/xc_meth_train.csv")
    # pd.DataFrame(xc_meth_val).to_csv("data/test_data/xc_meth_val.csv")
    pd.DataFrame(xc_meth_test).to_csv("MTEGDRP-main/data/test_data/xc_meth_test.csv")

    xc_mut_train = xc_mut[:size]
    xc_mut_val = xc_mut[size2:size1]
    xc_mut_test = xc_mut[size1:]
    # pd.DataFrame(xc_mut_train).to_csv("data/test_data/xc_mut_train.csv")
    # pd.DataFrame(xc_mut_val).to_csv("data/test_data/xc_mut_val.csv")
    pd.DataFrame(xc_mut_test).to_csv("MTEGDRP-main/data/test_data/xc_mut_test.csv")

    y_train = y[:size]
    y_val = y[size2:size1]
    y_test = y[size1:]
    # pd.DataFrame(y_train).to_csv("data/test_data/y_train.csv")
    # pd.DataFrame(y_val).to_csv("data/test_data/y_val.csv")
    pd.DataFrame(y_test).to_csv("MTEGDRP-main/data/test_data/y_test.csv")

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset + '_train_mix', xd=xd_train, xt_ge=xc_ge_train,
                                xt_meth=xc_meth_train, xt_mut=xc_mut_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset + '_val_mix', xd=xd_val, xt_ge=xc_ge_val,
                              xt_meth=xc_meth_val, xt_mut=xc_mut_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset + '_test_mix', xd=xd_test, xt_ge=xc_ge_test,
                               xt_meth=xc_meth_test, xt_mut=xc_mut_test, y=y_test, smile_graph=smile_graph)
    print("build data complete")

    return y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.KernelPCA, 1.PCA, 2.Isomap')
    args = parser.parse_args()
    choice = args.choice
    save_mix_drug_cell_matrix()