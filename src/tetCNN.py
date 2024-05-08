import os
import tet_mesh
import util
from scipy.io import loadmat
from torch_geometric import utils
from torch_geometric import transforms
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import segregate_self_loops,add_remaining_self_loops
import torch_geometric.transforms as T
import argparse
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import Net, saliency_map, grad_cam
from torch_geometric.data import Batch

from torch_geometric.nn import summary

def load_data(pos_folder, neg_folder, load=True, batch_size=8, cv=0):
    ###loading AD files and LBOs
    print('begin loading data.')
    if load:   # if run python tetCNN_LBO.py --load it will go through loading all
        pos_tet_list = []
        pos_LBO_list = []
        pos_mass_list = []

        neg_tet_list = []
        neg_LBO_list = []
        neg_mass_list = []

        pos_files_tet = util.list_files(pos_folder)
        for sample in pos_files_tet:
            pos_tet_list.append(tet_mesh.load_mesh(sample))

        pos_files_LBO = util.read_cot(pos_folder)
        for sample in pos_files_LBO:
            pos_LBO_list.append(util.load_LBO(sample, 'L').astype(np.float32))
        
        pos_files_mass = util.read_mass(pos_folder)
        for sample in pos_files_mass:
            pos_mass_list.append(util.load_LBO(sample, 'm').astype(np.float32))

        ############### CTL or any other groups
            
        neg_files_tet = util.list_files(neg_folder)
        for sample in neg_files_tet:
            neg_tet_list.append(tet_mesh.load_mesh(sample))

        neg_files_LBO = util.read_cot(neg_folder)
        for sample in neg_files_LBO:
            neg_LBO_list.append(util.load_LBO(sample, 'L').astype(np.float32))
        
        neg_files_mass = util.read_mass(neg_folder)
        for sample in neg_files_mass:
            neg_mass_list.append(util.load_LBO(sample, 'm').astype(np.float32))

        pos_list_tet_data = util.tet_data_LBO(pos_tet_list, pos_LBO_list, pos_mass_list, True)
        neg_list_tet_data = util.tet_data_LBO(neg_tet_list, neg_LBO_list, neg_mass_list, False)
        ###combinig two together
        list_tet_data = pos_list_tet_data + neg_list_tet_data #+ MCI_list_tet_data
        # util.save_data(list_tet_data, 'tetmesh_198_left.dat')

    else:
        print('here')
        list_tet_data = util.load_data('tetmesh_198_left.dat', n_lmk, BNN)


    ###############
    torch.manual_seed(42)
    random.shuffle(list_tet_data)
    
    if cv == 5:
        train_dataset = list_tet_data[:len(list_tet_data)*cv//5]
    else:
        train_dataset = list_tet_data[:len(list_tet_data)*cv//5] + list_tet_data[len(list_tet_data)*(cv+1)//5:]
    test_dataset = list_tet_data[len(list_tet_data)*cv//5:len(list_tet_data)*(cv+1)//5]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print('finish loading.')

    return train_loader, test_loader

    ##############


def train_(loader):
    with torch.no_grad():
        CM=0
        correct = 0
        train_loss = 0
        n = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out, _ = model(data=data, training=False)
            loss = criterion(out, data.y)
            train_loss += loss
            n += 1
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(data.y.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(data.y).sum().item()
        return CM, train_loss / n  #correct / len(loader.dataset)  # Derive ratio of correct predictions.

def train(loader):
    correct1 = 0
    train_loss = 0
    n = 0
    for data in loader:
        # data_list = data.to_data_list()
        data = data.to(device)
        out, _ = model(data=data, training=True)
        loss = criterion(out, data.y)
        train_loss += loss
        n += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / n

def test(loader):
    with torch.no_grad():
        CM=0
        correct = 0
        test_loss = 0
        n = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out, _ = model(data=data, training=False)
            loss = criterion(out, data.y)
            test_loss += loss
            n += 1
            #print(h.edge_index)
    #         cams = cam_extractor(out.squeeze(0).argmax().item(), out)
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(data.y.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(data.y).sum().item()
        return CM, out, test_loss / n #correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    # load_data(['/tetCNN/data/test/lh/ad'], ['/tetCNN/data/test/rh/ad'], ['/tetCNN/data/test/lh/cn'], ['/tetCNN/data/test/rh/cn'], True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', dest='pos')
    parser.add_argument('--neg', dest='neg')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--n_epoch', type=int, dest='n_epoch')
    parser.add_argument('--lr', type=float, default=0.001, dest='lr')
    parser.add_argument('--wd', type=float, default=0.0001, dest='wd')
    parser.add_argument('--batch_size', type=int, default=8, dest='batch_size')
    parser.add_argument('--cv', type=int, dest='cv')

    args, _ = parser.parse_known_args()

    train_loader, test_loader = load_data(args.pos, args.neg, args.load, args.batch_size, args.cv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    res_test_acc = []

    for epoch in range(1, args.n_epoch+1):
        train_loss = train(train_loader)
        train_acc, val_loss = train_(train_loader)
        test_acc, out, test_loss = test(test_loader)

        tn_train=train_acc[0][0]
        tp_train=train_acc[1][1]
        fp_train=train_acc[0][1]
        fn_train=train_acc[1][0]
        tr_acc = np.sum(np.diag(train_acc)/np.sum(train_acc))
        
        tn_test=test_acc[0][0]
        tp_test=test_acc[1][1]
        fp_test=test_acc[0][1]
        fn_test=test_acc[1][0]
        test_acc = np.sum(np.diag(test_acc)/np.sum(test_acc))
        test_sen=tp_test/(tp_test+fn_test)
        test_spe = tn_test/(tn_test+fp_test)
        test_prec=tp_test/(tp_test+fp_test)
        res_test_acc.append(test_acc)
        print('Epoch: {:02d}, Train loss: {:.4f}, Val loss: {:.4f}, Test loss: {:.4f}, Train acc: {:.4f}, Test acc: {:.4f}, Test sen: {:.4f}, Test spe: {:.4f}, Test prec: {:.4f}'.format(epoch, train_loss, val_loss, test_loss, tr_acc, test_acc,test_sen,test_spe,test_prec))

        # if epoch % 20 == 0:
        #     torch.save(model, os.path.join('/home/ychen855/tetCNN/checkpoints', str(args.cv) + '_' + str(epoch) + '.pth'))

    print('best test acccuracy: ', max(res_test_acc))

###### gradcam ######


    # optimizer.zero_grad()
    # X = next(iter(test_loader))
    # X.to(device)
    # out,h = model(X)
    # model.final_conv_acts.retain_grad()
    # loss = criterion(out, X.y)  # Compute the loss.
    # #print(loss)
    # loss.backward()  # Derive gradients.

    # # saliency_map_weights = saliency_map()
    # grad_cam_weights = grad_cam(model.final_conv_acts, model.final_conv_grads)
    # scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1))

    # from torch_geometric.nn import  knn_interpolate
    # tmp = torch.tensor(scaled_grad_cam_weights)
    # print(tmp.view(-1,1).cuda())
    # h.x = tmp.view(-1,1)
    # h = h.to(device)
    # print('dim h.x: ' + str(h.x.shape))
    # print('dim h.pos: ' + str(h.pos.shape))
    # print('dim x.pos: ' + str(X.pos.shape))
    # x = knn_interpolate(h.x, h.pos, X.pos)
    # # print('before savetxt')
    # # np.savetxt(os.path.join(os.path.abspath('.'), 'cams', 'gradcam_mci_cn.txt'),x.cpu(), fmt='%.6e')

