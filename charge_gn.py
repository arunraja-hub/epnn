import os
import scipy
import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
import torch_geometric.nn as geom_nn
import torch_geometric.loader as loader
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import TensorDataset, DataLoader
import sys
import wandb
wandb.init(mode="disabled")
wandb.login()


print('sys.argv[1]', sys.argv[1])

wandb_logger = WandbLogger(project=str(sys.argv[1]), 
                           name=str(sys.argv[1])+'pytorch_epnn_trial',
                           log_model='all',
                           save_dir='epnn_wandb_log/')


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

atom_num_dict = {'H' : 1,
             'C' : 6,
             'N' : 7,
             'O' : 8,
             'F' : 9,
             'P' : 15,
             'S' : 16,
             'Cl': 17,
             'Br': 35,
             }
elem_dict = {'H' : 0,
             'C' : 1,
             'N' : 2,
             'O' : 3,
             'F' : 4,
             'P' : 5,
             'S' : 6,
             'Cl': 7,
             'Br': 8,
             }

# class MLP_layer(tf.keras.layers.Layer):
#     def __init__(self, nodes, out_dim=1, activation='relu'):
#         super(MLP_layer, self).__init__()
#         self.nodes = nodes
#         self.layer_set = []
#         self.out_dim = out_dim
#         self.activation = activation
#         for num in nodes:
#             self.layer_set.append(Dense(num, activation=activation))
#         self.layer_set.append(Dense(out_dim, activation=None))

#     @tf.function(experimental_relax_shapes=True)
#     def call(self, x):
#         for layer in self.layer_set:
#             x = layer(x)
#         return x

class MLPModel(nn.Module):

    def __init__(self, nodes, out_dim=1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        self.nodes = nodes
        layers = []
        self.out_dim = out_dim

        for l in range(len(self.nodes)):
            layers += [
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
                # nn.Dropout(dp_rate)
            ]
            # in_channels = c_hidden
        layers += [nn.Linear(32, 1), 
                   nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv,
                     "GCN2": geom_nn.GCN2Conv, 'MPNN':geom_nn.MessagePassing}

class GNNModel(nn.Module):
    def __init__(
        self, T):
        super().__init__()
        self.T = T
        gnn_layer = geom_nn.GCNConv

        layers = []
        in_channels, out_channels = 48, 48
        # c_in, c_hidden
        for t in range(self.T):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels),
                # nn.LayerNorm(out_channels),
                # nn.LeakyReLU(inplace=True),
                # nn.Dropout(dp_rate),
            ]
        #     in_channels = c_hidden
        self.mpnn = nn.ModuleList(layers)
        print(layers)
        print(self.mpnn)

    def forward(self, h, e, x, q, mask):
        natom = e.shape[1]
        print('e', e.shape)
        print('x, h, q', x.shape, h.shape, q.shape)
        inp_atom_i = torch.concat([x, h, q], dim=-1)  # nmolec x natom x 9+32+1
        print('inp_atom_i', inp_atom_i.shape)
        inp_i = torch.tile(torch.unsqueeze(inp_atom_i, axis=1), [1, natom, 1,1]) # nmolec x natom x natom x 9+32+1
        # check if transposed correctly
        print('inp_i>>>>', inp_i.shape)
        inp_j = torch.transpose(inp_i, 1,2) #nmolec x natom x natom x 9+32+1
        print('inp_j>>>>', inp_j.shape)
        inp_ij = torch.concat([inp_i, inp_j, e], dim=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
        flat_inp_ij = torch.reshape(inp_ij, (-1, inp_ij.shape[-1]))#107584, 166
        
        for layer in self.mpnn:
            print('inp_i-for--', inp_i.shape)
            print('e -for--', e.shape)
            inp_i = layer(inp_atom_i, e.to(dtype=torch.int64) )
            print('inp_i-for--after layer-', inp_i.shape)
            print('inp_i-for--after layer-', e.shape)
        return inp_i
    


class EPNNModel(nn.Module):
    """Special 'Electron Passing Network,' which retains conservation of electrons but allows non-local passing"""

    def __init__(self, pass_fn, T=1):
        super(EPNNModel, self).__init__()
        self.pass_fns = []
        for t in range(T):
            self.pass_fns.append(pass_fn([32,32]))
        self.T = T

    # @tf.function(experimental_relax_shapes=True)
    def forward(self, h, e, x, q, mask):
        
        tol = 1e-5
        clip = torch.clamp(e, min=tol, max=1e5) #sets passed charges to be within 1e-5 to 1e5 range
        largest_e, largest_e_index = torch.max(clip, dim=-1)
        is_near = torch.not_equal(largest_e, tol)
        is_near = is_near.type(torch.float32)
        # tf.cast(is_near, dtype=tf.float32)

        natom = e.shape[1]
        mask = mask.type(torch.float32)
        # mask = tf.cast(mask, dtype=tf.float32)
        for t in range(self.T):
            self.pass_fn = self.pass_fns[t]

            print('_________________',x.shape, h.shape, q.shape)
            inp_atom_i = torch.concat([x, h, q], dim=-1)  # nmolec x natom x 9+32+1
            inp_i = torch.tile(torch.unsqueeze(inp_atom_i, dim=2), [1, 1, natom, 1]) # nmolec x natom x natom x 9+32+1
            inp_j = torch.transpose(inp_i, [0, 2, 1, 3]) #nmolec x natom x natom x 9+32+1
            inp_ij_N = torch.concat([inp_i, inp_j, e], dim=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32
            inp_ij_T = torch.concat([inp_j, inp_i, e], dim=-1) #nmolec x natom x natom x 9*2 + 32*2 + 1*2 + 32

            flat_inp_ij = torch.reshape(inp_ij_N, [-1, inp_ij_N.shape[-1]])
            flat_inp_ji = torch.reshape(inp_ij_T, [-1, inp_ij_T.shape[-1]])

            elec_ij_flat = self.pass_fn(flat_inp_ij)
            elec_ji_flat = self.pass_fn(flat_inp_ji)

            elec_ij = torch.reshape(elec_ij_flat, [-1, natom, natom])
            elec_ji = torch.reshape(elec_ji_flat, [-1, natom, natom])

            antisym_pass = 0.5 * (elec_ij - elec_ji) * torch.math.reduce_max(mask, axis=-1) * is_near

            q += torch.unsqueeze(torch.sum(antisym_pass, dim=2), dim=-1)
        return q


def get_init_edges(xyz, molecular_splits, num=32, cutoff=3.0, eta=2.0): #0.1 to 3 Angstrom
    mu = np.linspace(0.1, cutoff, num=num)
    D = scipy.spatial.distance_matrix(xyz,xyz)

    print('D shape', D.shape)
    print('molecular_splits.shape', molecular_splits.shape)

    # what is molec_vecA and molec_vecB?
    if molecular_splits.shape == (0,):
        adj = np.ones(D.shape)
    elif molecular_splits.shape == ():
        molec_vecA = np.zeros(D.shape[0])
        molec_vecA[:molecular_splits] = 1
        molec_vecB = np.zeros(D.shape[0])
        molec_vecB[molecular_splits:] = 1
        adj = np.outer(molec_vecA, molec_vecA.T) + np.outer(molec_vecB, molec_vecB.T)
    else:
        adj = np.zeros(D.shape)
        prev_split = 0
        for i, split in enumerate(molecular_splits):
            molec_vec = np.zeros(D.shape[0])
            molec_vec[prev_split:split] = 1
            # print(molec_vec)
            molec_mat = np.outer(molec_vec, molec_vec.T)
            print(molec_mat)
            # adj += molec_mat
        # print(adj)
        exit()
    adj = np.expand_dims(adj, -1)

    C = (np.cos(np.pi * (D - 0.0) / cutoff) + 1.0) / 2.0

    C[D >= cutoff] = 0.0
    C[D <= 0.0] = 1.0
    np.fill_diagonal(C, 0.0)
    D = np.expand_dims(D, -1)
    D = np.tile(D, [1, 1, num])
    C = np.expand_dims(C, -1)
    C = np.tile(C, [1, 1, num])
    mu = np.expand_dims(mu, 0)
    mu = np.expand_dims(mu, 0)
    mu = np.tile(mu, [D.shape[0], D.shape[1], 1])
    e = C * np.exp(-eta * (D-mu)**2)
    e = np.array(e, dtype=np.float32)

    return e, C

def gen_padded_init_state(path,  h_dim, e_dim):
    x = []
    h = []
    q = []
    Q = []
    e = []
    y = []
    soft_mask = []
    names = []

    for_loop_count = 0 
    for filename in os.listdir(path):
        label_file = path + filename[:-4] + '.npy'
        print('now processing for label_file-', label_file)
        #if os.path.exists(label_file) and filename.endswith(".xyz"):
        if filename.endswith(".xyz"):
            splits_path = path + filename[:-4] + "splits.npy"
            print('now finding if this splits_path exists-', splits_path)
            if os.path.exists(splits_path):
                splits = np.load(splits_path)
            else: 
                splits = np.array([])
            xyzfile = open(path + filename, 'r')
            lines = xyzfile.readlines()
            label_file = path + filename[:-4] + '.npy' #is this necessary-line 406?
            if os.path.exists(label_file):
                y.append(np.array(np.load(label_file), dtype=np.float32))
            else:
                print('No labels provided, y set to 0')
                y.append(np.zeros(len(lines)-2))
            Q.append(np.array(lines[1].strip().split()[0], dtype=np.float32))
            print('formal charges Q len', len(Q))
            names.append(filename[:-4])
            print('(filename[:-4] appended to names-', (filename[:-4]))
            xyz = []
            this_x = []
            for line in lines[2:]:
                data = line.split()
                elem_name = data[0]
                xyz.append([data[1], data[2], data[3]])
                ohe = np.zeros(len(elem_dict)+1)
                ohe[0] = atom_num_dict[elem_name]
                ohe[elem_dict[elem_name] + 1] = 1
                this_x.append(ohe)
            # this_x is a list of OHEs of atoms in molecule
            this_x = np.array(this_x)
            xyz = np.array(xyz, dtype=np.float32)
            these_edges, _ = get_init_edges(xyz, splits, num=e_dim) #soft_mask is not used
            print('these_edges -----', these_edges.shape)
            e.append(these_edges.reshape((-1, these_edges.shape[-1])))
            # soft_mask.append(this_soft_mask)

            print('this_x', this_x.shape)
            print('these_edges', these_edges.shape)
            # print('this_soft_mask', this_soft_mask.shape)
            # print('np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1))', np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1)).shape)

            x.append(np.tile(np.array(this_x, dtype=np.float32), (these_edges.shape[0], 1)))
            h.append(np.tile(np.zeros((this_x.shape[0], h_dim), dtype=np.float32), (these_edges.shape[0], 1)))
            avg_q = Q[-1] / len(this_x)
            q.append(np.tile(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float32), (these_edges.shape[0], 1)))

            # x.append(np.array(this_x, dtype=np.float32))
            # h.append(np.zeros((this_x.shape[0], h_dim), dtype=np.float32))
            # avg_q = Q[-1] / len(this_x)
            # q.append(np.array(np.ones((len(this_x), 1)) * avg_q, dtype=np.float32))

            for_loop_count += 1
            # break

    largest_system = np.max([y[i].shape[0] for i in range(len(y))])

    print('largest_system', largest_system)
    print('len(Q)', len(Q))

    x_padded = np.zeros((len(Q), largest_system,  x[0].shape[1]))
    h_padded = np.zeros((len(Q), largest_system, h[0].shape[1]))
    q_padded = np.zeros((len(Q), largest_system, q[0].shape[1]))
    e_padded = np.zeros((len(Q), largest_system, largest_system, e[0].shape[1]))
    # soft_mask_padded = np.zeros((len(Q), largest_system, largest_system, 1))
    y_padded = np.zeros((len(Q), largest_system, 1))
    pad_n = np.zeros((len(Q)))
    mask = np.zeros((len(Q), largest_system, largest_system))


    for i in range(x_padded.shape[0]):
        molec_size = np.sqrt(x[i].shape[0]).astype(np.int32)
        for j in range(y[i].shape[0]):
            y_padded[i][j] = y[i][j]
            x_padded[i][j] = x[i][j]
            h_padded[i][j] = h[i][j]
            q_padded[i][j] = q[i][j]
            #soft_mask_padded[i][j] = soft_mask[i][j]  # not used
            pad_n[i] = j #not used
        for j in range(molec_size):
            for k in range(molec_size):
                # x_padded[i][j][k] = x[i][j*molec_size + k]
                # h_padded[i][j][k] = h[i][j*molec_size + k]
                # q_padded[i][j][k] = q[i][j*molec_size + k]
                e_padded[i][j][k] = e[i][j*molec_size + k]
                mask[i][j][k] = 1

    return x_padded, h_padded, q_padded, e_padded, Q, y_padded, mask, np.array(names)

class EntireModel(pl.LightningModule):
    def __init__(self, layers, h_dim, T):
        super(EntireModel, self).__init__()
         # Saving hyperparameters
        self.save_hyperparameters()
        self.message_model = MLPModel
        self.update_fn = MLPModel(layers, out_dim=h_dim)
        self.graph_net = GNNModel(T)
        self.electron_model = MLPModel
        self.electron_net = EPNNModel(self.electron_model, T)

    def forward(self, x, h, q,e, y, mask ):
        graph_feats = self.graph_net(h, e, x, q, mask)                                     # nmol x natom x h_dim
        print('graph_feats------------', graph_feats.shape)
        q_pred =self.electron_net(graph_feats, e, x, q, mask)                             # nmol x natom x 1
        loss = torch.nn.MSELoss(y, q_pred)
        return q_pred, loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 0.00001)
        return optimizer

    def training_step(self, batch):
        xt, ht, qt,et, yt, maskt = batch
        _, loss = self.forward(xt, ht, qt,et, yt, maskt)
        # , mode="train")
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, batch_size= self.bs)
        return loss

    def validation_step(self, batch):
        xt, ht, qt,et, yt, maskt = batch
        _, loss = self.forward(xt, ht, qt,et, yt, maskt)
        # , mode="val")
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size= self.bs)

    def test_step(self, batch):
        xt, ht, qt,et, yt, maskt = batch
        _, loss = self.forward(xt, ht, qt,et, yt, maskt)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True,  batch_size= self.bs)

    def reset_parameters(self):
        def _reset_module_parameters(module):
            for layer in module.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif hasattr(layer, 'children'):
                    for child_layer in layer.children():
                        _reset_module_parameters(child_layer)

        _reset_module_parameters(self)

# def make_model(layers, h_dim, T, n_elems, natom):#mask, natom):                    # mask: nmol x natom
#     message_model = MLPModel
#     update_fn = MLPModel(layers, out_dim=h_dim)
#     graph_net = GNNModel(message_model, update_fn, T)
#     electron_model = MLPModel
#     electron_net = EPNNModel(electron_model, T=T)

#     h_inp = tf.keras.Input(shape=(natom, natom, h_dim), dtype='float32', name='h_inp')          # nmol x natom x natom x h_dim
#     e_inp = tf.keras.Input(shape=(natom, natom, h_dim), dtype='float32', name='e_inp')          # nmol x natom x natom x h_dim
#     x_inp = tf.keras.Input(shape=(natom, natom, n_elems), dtype='float32', name='x_inp')        # nmol x natom x natom x n_elems
#     q_inp = tf.keras.Input(shape=(natom, natom, 1), dtype='float32', name='q_inp')              # nmol x natom x natom x 1
#     mask_inp = tf.keras.Input(shape=(natom, natom, 1), dtype='float32', name='mask_inp')        # nmol x natom x natom x 1
    
#     h = tf.math.divide_no_nan(tf.math.reduce_sum(h_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
#     x = tf.math.divide_no_nan(tf.math.reduce_sum(x_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
#     q = tf.math.divide_no_nan(tf.math.reduce_sum(q_inp, axis=1), tf.math.reduce_sum(mask_inp, axis=1))
    
#     graph_feats = graph_net(h, e_inp, x, q, mask_inp)                                     # nmol x natom x h_dim
#     q_pred = electron_net(graph_feats, e_inp, x, q, mask_inp)                             # nmol x natom x 1

#     model = tf.keras.Model(inputs=[h_inp, e_inp, x_inp, q_inp, mask_inp], outputs=q_pred)

#     return model

# @tf.function(experimental_relax_shapes=True)
    
# def train_step(model, h, e, x, q, y, mask):
#     model.reset_parameters()
#     predictions = model([h, e, x, q, mask])
#     loss = torch.nn.MSELoss(y, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     train_loss(loss)
#     train_acc(predictions, y)
#     return predictions

# # @tf.function(experimental_relax_shapes=True)
# def test_step(h, e, x, q, y, mask):
#     predictions = model([h, e, x, q, mask])
#     t_loss = tf.keras.losses.MSE(y, predictions)
#     test_loss(t_loss)
#     test_acc(predictions, y)
#     return predictions

def train_test(model, train_loader, test_loader):
    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    CHECKPOINT_PATH = "epnn_saved_models/"

    root_dir = os.path.join(CHECKPOINT_PATH)
    # , "EPNN" + model_name)
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(dirpath=root_dir,save_weights_only=True, mode="min", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                        #  check_val_every_n_epoch=1,
                         max_epochs=500,
                         enable_progress_bar=True,
                         logger=wandb_logger) # False because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    trainer.fit(model, train_loader, test_loader)
    test_result = trainer.test(model, test_loader, verbose=True)
    batch = next(iter(train_loader))
    batch = batch.to(model.device)
    _, train_loss = model.forward(batch, mode="train")
    _, val_loss = model.forward(batch, mode="val")
    result = {"train": train_loss,
              "val": val_loss,
              "test": test_result[0]['test_loss']}
    return model, result

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train loss: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val loss:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test loss:  {(100.0*result_dict['test']):4.2f}%")


if __name__ == "__main__":
    # should be 32?
    h_dim = 48
    e_dim = 48
    layers = [32, 32]
    T = 5
    path = 'data/mixed/'
    # elem_dict has 9, anyway not used
    # n_elems = 10
    # optimizer = tf.keras.optimizers.Adam()
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_acc = tf.keras.metrics.MeanAbsoluteError(name='train_acc')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_acc = tf.keras.metrics.MeanAbsoluteError(name='test_acc')
    EPOCHS = 500
    best_test_acc = np.inf

    x, h, q, e, Q, y, mask, names = gen_padded_init_state(path, h_dim, e_dim)

# -----------------------
    # print('x, h, q, e, Q, y, mask, names', x, h, q, e, Q, y, mask, names)

    

    model = EntireModel(layers, h_dim, T)
    # .reset_parameters()
    # make_model(layers, h_dim, T, n_elems, x.shape[1])

    xt, xe, ht, he, qt, qe, et, ee, Qt, Qe, yt, ye, maskt, maske, namest, namese = train_test_split(x,h,q,e,Q,y,mask,names, test_size=0.2, random_state=42)
    
    print('xt, xe, ht, he, qt, qe, et, ee, Qt, Qe, yt, ye, maskt, maske, namest, namese ', 
          xt[0].shape, xe[0].shape, ht[0].shape, he[0].shape, qt[0].shape, qe[0].shape, et[0].shape, ee[0].shape, Qt[0].shape, Qe[0].shape, yt[0].shape, ye[0].shape, maskt[0].shape, maske[0].shape, namest[0].shape, namese[0].shape)
        #   type(xt), type(xe), type(ht), type(he), type(qt), type(qe), type(et), type(ee), type(Qt), type(Qe), type(yt), type(ye), type(maskt), type(maske), type(namest), type(namese) )

    train_tensor = TensorDataset(torch.tensor(xt),torch.tensor(ht),torch.tensor(qt),torch.tensor(et), torch.tensor(yt),torch.tensor(maskt))
    test_tensor = TensorDataset(torch.tensor(xe),torch.tensor(he),torch.tensor(qe),torch.tensor(ee), torch.tensor(ye),torch.tensor(maske))

    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True )
    test_loader = DataLoader(test_tensor, batch_size=64, shuffle=True)

    np.save("train_names.npy", namest, allow_pickle=True)
    np.save("val_names.npy", namese, allow_pickle=True)


    trained_model, result = train_test(model, train_loader, test_loader)
    print_results(result)

    # for epoch in range(EPOCHS):
    #     train_loss.reset_states()
    #     train_acc.reset_states()
    #     test_loss.reset_states()
    #     test_acc.reset_states()
    #     train_preds = []
    #     test_preds = []
    #     for i in range(len(xt)):
    #         hb = np.array(np.expand_dims(ht[i], axis=0))
    #         eb = np.array(np.expand_dims(et[i], axis=0))
    #         xb = np.array(np.expand_dims(xt[i], axis=0))
    #         qb = np.array(np.expand_dims(qt[i], axis=0))
    #         yb = np.array(np.expand_dims(yt[i], axis=0))
    #         maskb = np.array(np.expand_dims(maskt[i], axis=0))
    #         #train_preds.append(train_step(ht[i], et[i], xt[i], qt[i], yt[i], maskt[i]))
    #         train_preds.append(train_step(hb, eb, xb, qb, yb, maskb))
    #     for i in range(len(xe)):
    #         hb = np.array(np.expand_dims(he[i], axis=0))
    #         eb = np.array(np.expand_dims(ee[i], axis=0))
    #         xb = np.array(np.expand_dims(xe[i], axis=0))
    #         qb = np.array(np.expand_dims(qe[i], axis=0))
    #         yb = np.array(np.expand_dims(ye[i], axis=0))
    #         maskb = np.array(np.expand_dims(maske[i], axis=0))
    #         test_preds.append(test_step(hb, eb, xb, qb, yb, maskb))
    #     if test_acc.result() < best_test_acc:
    #         best_test_acc = test_acc.result()
    #         model.save_weights('models/model_weights')
    #     #    model.save(f'models/model')
    #     #    #tf.saved_model.save(model(he[-1], ee[-1], xe[-1], qe[-1]), 'models/model')
    #         np.save("train_pred_charges.npy", np.array(np.squeeze(train_preds)))
    #         np.save("train_lab_charges.npy", np.squeeze(yt))
    #         np.save("test_pred_charges.npy", np.array(np.squeeze(test_preds)))
    #         np.save("test_lab_charges.npy", np.squeeze(ye))

    #     template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
    #     print(template.format(epoch, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()))
