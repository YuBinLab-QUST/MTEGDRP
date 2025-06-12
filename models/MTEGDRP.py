"""
__coding__: utf-8
__Author__: Liu Zhihan
__Time__: 2024/8/21 14:30
__File__: MTEGDRP.py
__remark__:
__Software__: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv as GIN_layer
from torch_geometric.nn import GCNConv as GCN_layer
from torch_geometric.nn import GATConv as GAT_layer, BatchNorm, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch import nn, einsum
from einops import rearrange
from egnn_pytorch import EGNN
from transformers import AutoTokenizer, AutoModelForMaskedLM

DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim=-1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=78, heads=6, dim_head=13, Lg=0.5, Ld=0.5, La=1, dist_kernel_fn='exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h=h, qkv=3).unbind(dim=-2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim=-1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MAT(nn.Module):
    def __init__(
            self,
            *,
            dim_in=78,
            model_dim=78,
            dim_out=78,
            depth=1,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, Attention(model_dim, heads=heads, Lg=Lg, Ld=Ld, La=La,
                                                      dist_kernel_fn=dist_kernel_fn))),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)

    def forward(
            self,
            x,
            mask=None,
            adjacency_mat=None,
            distance_mat=None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask=mask,
                adjacency_mat=adjacency_mat,
                distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x



# Define the Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2048, d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


# Define the Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)


class MTEGDRP(torch.nn.Module):
    def __init__(self, output_dim=1, num_features_xd=78,
                 ge_features_dim=128, num_features_xt=25, embed_dim=128,
                 mut_feature_dim=128, meth_feature_dim=128, connect_dim=128, dropout=0.2):
        super(MTrsDRP, self).__init__()

        self.mat = MAT(
            dim_in=78,
            model_dim=78,
            dim_out=78 * 2,
            depth=4,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp')
        self.conv_gcn = GCN_layer(num_features_xd * 2, num_features_xd * 2)

        net1 = Sequential(Linear(num_features_xd * 2, num_features_xd * 2), ReLU(),
                          Linear(num_features_xd * 2, num_features_xd * 2))
        self.conv_gin1 = GIN_layer(net1)
        self.bn1 = torch.nn.BatchNorm1d(num_features_xd * 2)

        self.drug_layer1 = EGNN(dim=78)
        self.drug_layer2 = EGNN(dim=78)
        self.drug_layer3 = EGNN(dim=156) 
        self.fc_drug_jihe1 = Linear(78,390)
        self.fc_drug_jihe2 = Linear(390,156)

        self.fc3_drug = Linear(312,156)

        self.fc1_drug = Linear(num_features_xd * 6, num_features_xd * 12)
        self.fc2_drug = Linear(num_features_xd * 12,num_features_xd * 6)


        # 单组组学数据特征--GE
        self.EncoderLayer_ge_1 = nn.TransformerEncoderLayer(d_model=ge_features_dim, nhead=1, dropout=0.5)
        self.conv_ge_1 = nn.TransformerEncoder(self.EncoderLayer_ge_1, 1)
        self.EncoderLayer_ge_2 = nn.TransformerEncoderLayer(d_model=ge_features_dim, nhead=1, dropout=0.5)
        self.conv_ge_2 = nn.TransformerEncoder(self.EncoderLayer_ge_2, 1)
        self.EncoderLayer_ge_3 = nn.TransformerEncoderLayer(d_model=ge_features_dim, nhead=1, dropout=0.5)
        self.conv_ge_3 = nn.TransformerEncoder(self.EncoderLayer_ge_3, 1)
        self.EncoderLayer_ge_4 = nn.TransformerEncoderLayer(d_model=ge_features_dim, nhead=1, dropout=0.5)
        self.conv_ge_4 = nn.TransformerEncoder(self.EncoderLayer_ge_4, 1)
        self.fc1_ge = Linear(ge_features_dim, 4000)
        self.fc2_ge = Linear(4000, connect_dim)

        # 单组组学数据特征--MUT
        self.EncoderLayer_mut_1 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_1 = nn.TransformerEncoder(self.EncoderLayer_mut_1, 1)
        self.EncoderLayer_mut_2 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_2 = nn.TransformerEncoder(self.EncoderLayer_mut_2, 1)
        self.EncoderLayer_mut_3 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_3 = nn.TransformerEncoder(self.EncoderLayer_mut_3, 1)
        self.EncoderLayer_mut_4 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_4 = nn.TransformerEncoder(self.EncoderLayer_mut_4, 1)
        self.fc1_mut = Linear(mut_feature_dim, 4000)
        self.fc2_mut = Linear(4000, connect_dim)

        # 单组组学数据特征--METH
        self.EncoderLayer_meth_1 = nn.TransformerEncoderLayer(d_model=meth_feature_dim, nhead=1, dropout=0.5)
        self.conv_meth_1 = nn.TransformerEncoder(self.EncoderLayer_meth_1, 1)
        self.EncoderLayer_meth_2 = nn.TransformerEncoderLayer(d_model=meth_feature_dim, nhead=1, dropout=0.5)
        self.conv_meth_2 = nn.TransformerEncoder(self.EncoderLayer_meth_2, 1)
        self.EncoderLayer_meth_3 = nn.TransformerEncoderLayer(d_model=meth_feature_dim, nhead=1, dropout=0.5)
        self.conv_meth_3 = nn.TransformerEncoder(self.EncoderLayer_meth_3, 1)
        self.EncoderLayer_meth_4 = nn.TransformerEncoderLayer(d_model=meth_feature_dim, nhead=1, dropout=0.5)
        self.conv_meth_4 = nn.TransformerEncoder(self.EncoderLayer_meth_4, 1)
        self.fc1_meth = Linear(meth_feature_dim, 4000)
        self.fc2_meth = Linear(4000, connect_dim)

        # Define the Transformer Decoder
        self.decoder = TransformerDecoder(
            num_layers=4,
            d_model=852,  # Concatenated feature dimensions
            nhead=4,  # Number of attention heads
            dropout=0.1
        )

        self.fc1_all = Linear(852, 1024)
        self.fc2_all = Linear(1024, 512)
        self.fc3_all = Linear(512, 256)
        self.fc4_all = Linear(256, 128)
        self.out = Linear(connect_dim, output_dim)

        # 激活函数和正则化
        self.relu = ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        drug_poi_data, drug_edg_index, batch = data.x, data.edge_index, data.batch
        ge_data, meth_data, mut_data = data.target_ge, data.target_meth, data.target_mut
        coors = data.coordinates
        
        drug_data = torch.unsqueeze(drug_poi_data, 1)
        drug_data = self.mat(drug_data)
        drug_data = self.conv_gcn(drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)
        drug_data = F.relu(self.conv_gin1(drug_data, drug_edg_index))
        drug_data = self.bn1(drug_data)
        drug_data = self.conv_gcn(drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)
        
        egnn_list =[]
        index = 0
        for data_len in data.c_size:
            temp_data = drug_poi_data[index:index+data_len].unsqueeze(0)
            temp_coors = coors[index:index+data_len].unsqueeze(0)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer1(temp_data, temp_coors)
            temp_drug_data_jihe, temp_ehnncoors = self.drug_layer2(temp_drug_data_jihe,temp_ehnncoors)
            temp_drug_data_jihe = temp_drug_data_jihe.squeeze(0)
            egnn_list.append(temp_drug_data_jihe)
            index+=data_len
        
        egnn_features = torch.cat(egnn_list,dim=0)

        drug_data_final = torch.cat((drug_data,egnn_features),dim=1)
        drug_data_final = torch.cat([gmp(drug_data_final, batch), gap(drug_data_final, batch)], dim=1)
    
        
        drug_data = self.relu(self.fc1_drug(drug_data_final))
        drug_data = self.dropout(drug_data)
        drug_data = self.fc2_drug(drug_data)

        ge_data = ge_data[:, None, :]
        ge_data = self.conv_ge_1(ge_data)
        ge_data = self.conv_ge_2(ge_data)
        ge_data = self.conv_ge_3(ge_data)
        #ge_data = self.conv_ge_4(ge_data)
        ge_data = ge_data.view(-1, ge_data.shape[1] * ge_data.shape[2])
        ge_data = self.fc1_ge(ge_data)
        ge_data = self.dropout(self.relu(ge_data))
        ge_data = self.fc2_ge(ge_data)

        mut_data = mut_data[:, None, :]
        mut_data = self.conv_mut_1(mut_data)
        mut_data = self.conv_mut_2(mut_data)
        mut_data = self.conv_mut_3(mut_data)
        #mut_data = self.conv_mut_4(mut_data)
        mut_data = mut_data.view(-1, mut_data.shape[1] * mut_data.shape[2])
        mut_data = self.fc1_mut(mut_data)
        mut_data = self.dropout(self.relu(mut_data))
        mut_data = self.fc2_mut(mut_data)

        meth_data = meth_data[:, None, :]
        meth_data = self.conv_meth_1(meth_data)
        meth_data = self.conv_meth_2(meth_data)
        meth_data = self.conv_meth_3(meth_data)
        #meth_data = self.conv_meth_4(meth_data)
        meth_data = meth_data.view(-1, meth_data.shape[1] * meth_data.shape[2])
        meth_data = self.fc1_meth(meth_data)
        meth_data = self.dropout(self.relu(meth_data))
        meth_data = self.fc2_meth(meth_data)
        concat_data = torch.cat((drug_data, ge_data, meth_data, mut_data), 1)
        # Pass the concatenated features through the Transformer Decoder
        concat_data = concat_data.unsqueeze(0)  # Add batch dimension for Transformer input
        concat_data = self.decoder(concat_data, concat_data)
        concat_data = concat_data.squeeze(0)  # Remove batch dimension after processing



        # 隐藏层
        concat_data = self.fc1_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc2_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc3_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc4_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        out = self.out(concat_data)
        #out = self.sigmoid(out)
        #out = nn.Sigmoid()(out)
        return out, drug_data, mut_data