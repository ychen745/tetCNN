import math
import torch
import random
import numpy as np
from scipy.sparse.linalg import eigs   
from torch_geometric import utils
from typing import Optional
from torch_scatter import scatter
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops
from torch_sparse import coalesce
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import remove_self_loops
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool
import torch_geometric.nn as nn
from torch.nn import Linear
import os.path as osp
from torch_geometric.nn.pool import graclus
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import graclus, global_mean_pool
from torch_geometric.utils import segregate_self_loops,add_remaining_self_loops

import torch.nn as tnn

import util

def uniform(size, tensor):
    torch.manual_seed(42)
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    torch.manual_seed(42)
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    torch.manual_seed(42)
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def saliency_map(input_grads):
    print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]):
        node_grads = input_grads[n, :]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return node_saliency_map

def grad_cam(final_conv_acts, final_conv_grads):
    print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0)
    for n in range(final_conv_acts.shape[0]):
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map



####downsampling and pooling

def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def pool_edge(cluster, edge_index, edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        #print("passed")
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr

def pool_LBO(cluster, edge_index,
              edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    LBO_index2,LBO_weight2,LBO_index2_loop,LBO_weight2_loop =  segregate_self_loops(edge_index, edge_attr)

    edge_index = torch.cat((LBO_index2,LBO_index2_loop),dim = 1)
    edge_attr = torch.cat(((LBO_weight2),(LBO_weight2_loop)))
    #edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        #print("passed")
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr

def pool_batch(perm, batch):
    return batch[perm]

def pool_pos(cluster, pos):
    return scatter_mean(pos, cluster, dim=0)

def update_lmk(cluster, lmk_idx):
    new_lmk = scatter_add(lmk_idx, cluster, dim=0)
    new_lmk = torch.where(new_lmk > 0, 1, 0)
    return new_lmk


def _avg_pool_x(cluster, x, size: Optional[int] = None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')


def avg_pool_x(cluster, x, batch, size: Optional[int] = None):
    r"""Average pools node features according to the clustering defined in
    :attr:`cluster`.
    See :meth:`torch_geometric.nn.pool.max_pool_x` for more details.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): The maximum number of clusters in a single
            example. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`LongTensor`) if :attr:`size` is
        :obj:`None`, else :class:`Tensor`
    """
    if size is not None:
        batch_size = int(batch.max().item()) + 1
        return _avg_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _avg_pool_x(cluster, x)
    batch = pool_batch(perm, batch)

    return x, batch


def avg_pool(cluster, data, transform=None):
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    Final node features are defined by the *average* features of all nodes
    within the same cluster.
    See :meth:`torch_geometric.nn.pool.max_pool` for more details.

    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)
    lmax = data.lmax
    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    mass = None if data.mass is None else _avg_pool_mass(cluster, data.mass)
    index, attr = pool_edge(cluster, data.edge_index, data.edge_attr)
    LBO_index, LBO_weight = pool_LBO(cluster, data.LBO_index, data.LBO_weight)
    batch = None if data.batch is None else pool_batch(perm, data.batch)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    
#     data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos,LBO_index = LBO_index, LBO_weight=LBO_weight,mass = mass,lmax = lmax)
    data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos,LBO_index = LBO_index, LBO_weight=LBO_weight,lmax = lmax, mass = mass)

    if transform is not None:
        data = transform(data)

    return data


def _avg_pool_mass(cluster, x, size: Optional[int] = None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='sum')


    ################chebconv

from typing import Optional
from torch_geometric.typing import OptTensor
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

#from ..inits import glorot, zeros


class ChebConv2(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, K, normalization='sym',
                 bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv2, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        torch.manual_seed(42)
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(42)
        glorot(self.weight)
        zeros(self.bias)

    def __norm__(self,LBO_index,LBO_weight,
                  num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,

                 batch: OptTensor = None):

        edge_index, edge_weight = LBO_index, LBO_weight

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / -lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(self, x, edge_index,LBO_index,LBO_weight, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(LBO_index,LBO_weight, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                        batch=batch)
    
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


############### final network TetCNN
def normalized_cut_2d(edge_index, pos, LBO_weight, LBO_index, mass):
    row, col = edge_index
#     edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    edge_attr = torch.abs(LBO_weight[0:-len(pos)])
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class VisulizeAttention(torch.nn.Module):
    def __init__(self):
        super(VisulizeAttention,self).__init__()
        self.mid1 = torch.nn.Linear(64,32)
        self.tanh = torch.nn.Tanh()
        self.mid2 = torch.nn.Linear(32,1)
        
    def forward(self, x):
        x_1 = self.mid1(x)
        x_1 = self.tanh(x_1)
        x_2 = self.mid2(x_1)

        attention_map = torch.sigmoid(x_2)
        New_featureMap = x * attention_map
        
        return New_featureMap,attention_map
    
    
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        torch.manual_seed(42)
        self.conv1 = ChebConv2(3, 16, 1)
        self.conv2 = ChebConv2(16, 32, 1)
        #self.conv2 = nn.DataParallel(self.conv2)

        self.conv3 = ChebConv2(32, 64, 1)
        self.conv4 = ChebConv2(64, 128, 1)
        # self.conv5 = ChebConv2(128,256,1)
        
        self.attention = VisulizeAttention()
        self.fc1 = torch.nn.Linear(128,32)
        self.fc2 = torch.nn.Linear(32,2)
        # self.fc3 = torch.nn.Linear(32,2)
        self.bn1 = nn.BatchNorm(16)
        self.bn2 = nn.BatchNorm(32)
        self.bn3 = nn.BatchNorm(64)
        self.bn4 = nn.BatchNorm(128)
        self.bn5 = nn.BatchNorm(256)
        self.bn6 = nn.BatchNorm(32)
        self.weight = Parameter(torch.Tensor(64,64))
        glorot(self.weight)
        self.weight2 = Parameter(torch.Tensor(128,128))
        glorot(self.weight2)
        self.lin = Linear(64, 2)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, training):
        self.input = data.x
        self.training = training
        LBO_index_0 = data.LBO_index

        # data.x = F.relu(self.conv1(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax))
        data.x = F.relu(self.bn1(self.conv1(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax)))
        edge_weight = normalized_cut_2d(data.edge_index, data.pos, data.LBO_weight, data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, edge_weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))
        edge_weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, edge_weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.dropout(data.x, p = 0.2, training = self.training)

        # data.x = F.relu(self.conv2(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax))
        data.x = F.relu(self.bn2(self.conv2(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax)))
        weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))
        edge_weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, edge_weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.dropout(data.x, p = 0.2, training = self.training)

        # data.x = F.relu(self.conv3(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax))
        data.x = F.relu(self.bn3(self.conv3(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax)))
        weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))
        edge_weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, edge_weight, data.x.size(0))
        data.edge_attr = None
        data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))

        # data.x = F.relu(self.bn4(self.conv4(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax)))
        # weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        # cluster = graclus(data.edge_index, weight, data.x.size(0))
        # data.edge_attr = None
        # data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))
        # edge_weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        # cluster = graclus(data.edge_index, edge_weight, data.x.size(0))
        # data.edge_attr = None
        # data = avg_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.dropout(data.x, p = 0.2, training = self.training)

        with torch.enable_grad():
            self.final_conv_acts  = self.conv4(data.x, data.edge_index, data.LBO_index,data.LBO_weight,batch = data.batch,lambda_max = data.lmax)
        self.final_conv_acts.register_hook(self.activations_hook)
        self.final_conv_acts.retain_grad()

        data.x = F.relu(self.final_conv_acts)
        data.x = F.relu(self.bn4(self.final_conv_acts))
        h = data
        
        weight = normalized_cut_2d(data.edge_index, data.pos,data.LBO_weight,data.LBO_index, data.mass)
        cluster = graclus(data.edge_index, weight, data.x.size(0))

        x, batch = avg_pool_x(cluster, data.x, data.batch)
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))

        return self.fc2(x), h

