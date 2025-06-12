import os
import math
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, roc_auc_score


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt_ge=None, xt_meth=None, xt_mut=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, saliency_map=False, test_drug_dict=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt_ge, xt_meth, xt_mut, y, smile_graph, test_drug_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt_ge, xt_meth, xt_mut, y, smile_graph, test_drug_dict):
        if y is not None:
            assert (len(xd) == len(xt_ge) and len(xt_ge) == len(y)) and len(y) == len(xt_meth) and len(xt_meth) == len(
            xt_mut), "The four lists must be the same length!"
        else:
            assert len(xd) == len(xt_ge) == len(xt_meth) == len(xt_mut)
        data_list = []

        data_len = len(xd)
        print(data_len)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target_ge = xt_ge[i]
            target_meth = xt_meth[i]
            target_mut = xt_mut[i]
            labels = y[i] if y is not None else -1
            labels = np.array(labels).astype(float)

            # 从SMILES图结构中提取特征
            c_size, features, edge_index,coordinates = smile_graph[smiles]

            # 构建GCNData 对象
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                # y=torch.FloatTensor([labels]))
                y=torch.FloatTensor(labels),
                coordinates=torch.FloatTensor(coordinates),
                smiles = smiles
            )

            if self.saliency_map == True:
                GCNData.target_ge = torch.tensor([target_ge], dtype=torch.float, requires_grad=True)
                GCNData.target_meth = torch.tensor([target_meth], dtype=torch.float, requires_grad=True)
                GCNData.target_mut = torch.tensor([target_mut], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target_ge = torch.FloatTensor([target_ge])
                GCNData.target_meth = torch.FloatTensor([target_meth])
                GCNData.target_mut = torch.FloatTensor([target_mut])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            #GCNData.smiles = smiles
            
            data_list.append(GCNData)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd

def mae(y, f):
    mae = np.mean(np.abs(y-f))
    return mae

def rmse(y, f):
    rmse = sqrt(mean_squared_error(y, f))
    return rmse


def mse(y, f):
    mse = mean_squared_error(y, f)
    return mse


def r2(y, f):
    r2 = r2_score(y, f)
    return r2


def mse_cust(y, f):
    mse = ((y - f) ** 2)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def draw_cust_mse(mse_dict):
    best_mse = []
    best_mse_title = []
    i = 0
    for (key, value) in mse_dict.items():
        if i < 10 or (i > 13 and i < 24):
            best_mse.append(value)
            best_mse_title.append(key)
        i += 1

    plt.bar(best_mse_title, best_mse)
    plt.xticks(rotation=90)
    plt.title('GE & METH')
    plt.ylabel('MSE')
    plt.savefig("Blind drug.png")


def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(title + ".png")  # should before show method


def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    plt.savefig(title + ".png")  # should before show method