import os
from scipy.io import loadmat
from torch_geometric import utils
from torch_geometric import transforms
import torch
from torch_geometric.data import Data, HeteroData, DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import segregate_self_loops,add_remaining_self_loops
from torch_geometric.utils import to_undirected
import pickle
import numpy as np
from typing import Optional, Tuple


class TetraToEdge(object):
    r"""Converts mesh tetras :obj:`[4, num_tetras]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`tetra_to_edge`).
    Args:
        remove_tetras (bool, optional): If set to :obj:`False`, the tetra tensor
            will not be removed.
    """

    def __init__(self, remove_tetras: bool = True):
        self.remove_tetras = remove_tetras

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'tetra'):
            tetra = data.tetra
            edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_tetras:
                data.tetra = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def list_files(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            for f in files:
                if f.endswith('.node'):
                    flist.append(os.path.join(folder, subfolder, f))
    return flist

def read_LBO(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'LBO.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'LBO.mat'))
    return flist

def read_mass(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'mass.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'mass.mat'))
    return flist

def read_cot(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'cot.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'cot.mat'))
    return flist

def load_LBO(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containing the field name (default='shape')
    """
    return loadmat(path_file)[name_field]



def load_data(datafile):
    try:
        with open(datafile,"rb") as f:
            return pickle.load(f)
    except:
        x = []
    

def save_data(data, datafile):
    with open(datafile, "wb") as f:
        pickle.dump(data, f)


def from_pymesh(mesh):
    r"""Converts a :pymesh file to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        mesh (pymesh): A :obj:`pymesh` mesh.
    """

    pos = torch.from_numpy(mesh.vertices).to(torch.float)
    tetra = torch.from_numpy(mesh.voxels).to(torch.long).t().contiguous()

    return Data(pos=pos, tetra=tetra)

####LBO dataloader
def tet_data_LBO(list_tet, list_LBO, list_mass, positive):
    list_tet_data = []
    counter = 0
    l = len(list_tet)
    # l = 5

    for i in range (l):
        t2e = TetraToEdge()
        data = from_pymesh(list_tet[i])
        L = list_LBO[i]
        m = list_mass[i].todense()

        lambda_max = eigs(L, k=1, which='LM', return_eigenvectors=False)
        data = t2e(data)

        norm_pos = data.pos - torch.mean(data.pos, axis=0)
        norm_pos /= torch.max(torch.linalg.norm(norm_pos, axis=1))

        edge_index_L, edge_weight_L = from_scipy_sparse_matrix(L)

        LBO_index2, LBO_weight2, LBO_index2_loop, LBO_weight2_loop =  segregate_self_loops(edge_index_L, edge_weight_L)

        LBO_index = torch.cat((LBO_index2, LBO_index2_loop),dim = 1)
        LBO_weight = torch.cat((-torch.abs(LBO_weight2),torch.abs(LBO_weight2_loop)))
        LBO_index[[0,1]] =  LBO_index[[1,0]]
        data.lambda_max = float(lambda_max.real)

        if positive:
            data = Data(x = norm_pos, edge_index = data.edge_index, y = torch.tensor(1), pos = norm_pos,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(m), lmax = data.lambda_max)
        else:
            data = Data(x = norm_pos, edge_index = data.edge_index,y = torch.tensor(0), pos = norm_pos,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(m), lmax = data.lambda_max)

        list_tet_data.append((data))

    return list_tet_data




if __name__ == '__main__':
    folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/ad'

    tetlist = list_files(folder)
    lbolist = read_LBO(folder)
    masslist = read_mass(folder)
    cotlist = read_cot(folder)
    
    print(len(tetlist))
    print(len(lbolist))
    print(len(masslist))
    print(len(cotlist))