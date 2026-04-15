import os
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
"""
transform graphs (represented by edge list) to hypergraph (represented by node_dict & edge_dict)
"""
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as cos_dis
import torch

import warnings
from typing import Any

import torch

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class HyperGraphData(Data):
    r"""
        x (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Hyperedge tensor
            with shape :obj:`[2, num_edges*num_nodes_per_edge]`.
            Where `edge_index[1]` denotes the hyperedge index and
            `edge_index[0]` denotes the node indicies that are connected
            by the hyperedge. (default: :obj:`None`)
            (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`.
            (default: :obj:`None`)
        y (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos, **kwargs)

    @property
    def num_edges(self) -> int:
        r"""Returns the number of hyperedges in the hypergraph.
        """
        if self.edge_index is None:
            return 0
        return max(self.edge_index[1]) + 1

    @property
    def num_nodes(self) -> int:
        num_nodes = super().num_nodes
        if (self.edge_index is not None and num_nodes == self.num_edges):
            return max(self.edge_index[0]) + 1
        return num_nodes

    def is_edge_attr(self, key: str) -> bool:
        val = super().is_edge_attr(key)
        if not val and self.edge_index is not None:
            return key in self and self[key].size(0) == self.num_edges

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index':
            return torch.tensor([[self.num_nodes], [self.num_edges]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


    def has_isolated_nodes(self) -> bool:
        if self.edge_index is None:
            return False
        return torch.unique(self.edge_index[0]).size(0) < self.num_nodes


    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if 'edge_index' in self:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                status = False
                warn_or_raise(
                    f"'edge_index' needs to be of shape [2, num_edges] in "
                    f"'{cls_name}' (found {self.edge_index.size()})",
                    raise_on_error)

        if 'edge_index' in self and self.edge_index.numel() > 0:
            if self.edge_index.min() < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min())})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index[0].max() >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max())})", raise_on_error)

        return status


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)



def create_hyper_edges_from_matrix(matrix, k=5):
    hyper_edge_index = torch.zeros([2, matrix.shape[0] * (k+1)], dtype=torch.long)
    for node in range(matrix.shape[0]):
        # Get k closest nodes (include the node itself)
        connected_nodes = np.argpartition(
            matrix[node, :], -k-1)[-k-1:]
        # Assign each node its own hyperedge ID and associate its nearest neighbors
        for idx, connected_node in enumerate(connected_nodes):
            # Hyper edge's node
            hyper_edge_index[0, node*(k+1)+idx] = connected_node
            # Hyperedge ID
            hyper_edge_index[1, node*(k+1)+idx] = node

    return hyper_edge_index


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        ##
        self.num_tasks = 1
        self.task_type = 'classification'
        self.eval_metric = 'accuracy'

        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        ##
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # /home/zhang_istbi/zhangsj/gcn_project/A-GCL-main/ABIDE_116_itv/raw/ASD/sub-0050002_all_positive.mat
        
        
        path1 = osp.join(self.raw_dir, 'ASD')
        path1_nf = osp.join(self.raw_dir, 'ASD_node_feature')

        files = os.listdir(path1)  
        files_nf_ASD = os.listdir(path1_nf)

        data_list_ASD = []
        for file in files:  #'sub-29675_aal1_all_positive.mat'
            hyper_edge_index=[]
            if 'positive' in file:

                
                digit_filter = filter(str.isdigit, file)
                digit_list = list(digit_filter)
                #digit_str = "".join(digit_list[:-1])
                digit_str = "".join(digit_list[:-1])#29675
                
                for file_nf in files_nf_ASD:
                    if digit_str in file_nf:
                        # print("file_nf",file_nf)
                        # print("file_nf_path",osp.join(path1_nf, file_nf))
                        asd_nf_path=osp.join(path1_nf, file_nf)
                        print("asd_nf_path",asd_nf_path)
                        nf = sio.loadmat(asd_nf_path)  
                        #nf = sio.loadmat(osp.join(path1_nf, file_nf)) 
                        x = nf['alff_value_cache']
                        x = np.nan_to_num(x)
                        x = torch.Tensor(x)
                asd_adj_path=osp.join(path1, file)
                print("asd_adj_path",asd_adj_path)
                adj = sio.loadmat(asd_adj_path)#
                #adj = sio.loadmat(osp.join(path1, file))#29675
                edge_index = adj['corr_each_sub']
                edge_index = np.nan_to_num(edge_index) #pearson
                edge_index = torch.Tensor(edge_index)  
                pos = torch.diag(edge_index)
                fc_hyper_edges = create_hyper_edges_from_matrix(edge_index, k=5)#2*696
                hyper_edge_index.append(fc_hyper_edges)
                # convert hyper_edge_index to torch tensor
                hyper_edge_index = torch.cat(hyper_edge_index, dim=1)
                edge_weight = torch.ones(hyper_edge_index[1].max() + 1)           
                data = HyperGraphData(x=x, edge_index=hyper_edge_index,edge_weight=edge_weight,
                        pos=pos, y=0) #HyperGraphData(x=[116, 3], edge_index=[2, 696], y=0, pos=[116], edge_weight=[116])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list_ASD.append(data)
        
        path2 = osp.join(self.raw_dir, 'HC')
        path2_nf = osp.join(self.raw_dir, 'HC_node_feature')

        files = os.listdir(path2)  
        files_nf_TC = os.listdir(path2_nf)

        data_list_TC = []
        for file in files: 
            hyper_edge_index=[]
            if 'positive' in file:
                digit_filter = filter(str.isdigit, file)
                digit_list = list(digit_filter)
                #digit_str = "".join(digit_list[:-3])
                digit_str = "".join(digit_list[:-1])
                for file_nf in files_nf_TC:
                    if digit_str in file_nf:
                        tc_nf_path=osp.join(path2_nf, file_nf)
                        print("tc_nf_path",tc_nf_path)
                        nf = sio.loadmat(tc_nf_path) 
                        #nf = sio.loadmat(osp.join(path2_nf, file_nf)) 
                        x = nf['alff_value_cache']
                        x = np.nan_to_num(x)
                        x = torch.Tensor(x)
                tc_adj_path=osp.join(path2, file)
                print("tc_adj_path",tc_adj_path)
                adj = sio.loadmat(tc_adj_path)  
                edge_index = adj['corr_each_sub']
                edge_index = np.nan_to_num(edge_index)
                edge_index = torch.Tensor(edge_index)           
                pos = torch.diag(edge_index)
                fc_hyper_edges = create_hyper_edges_from_matrix(edge_index, k=5)
                hyper_edge_index.append(fc_hyper_edges)
                # convert hyper_edge_index to torch tensor
                hyper_edge_index = torch.cat(hyper_edge_index, dim=1)
                edge_weight = torch.ones(hyper_edge_index[1].max() + 1)                
                data = HyperGraphData(x=x, edge_index=hyper_edge_index,edge_weight=edge_weight,
                        pos=pos, y=1)
                

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list_TC.append(data)
        print("tc:", len(data_list_TC))
        print("tc:", len(data_list_ASD))
        data_list = data_list_ASD + data_list_TC
        print("all:", len(data_list))
       
        torch.save(self.collate(data_list),
                osp.join(self.processed_dir, 'data.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class TUEvaluator:
    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = 'accuracy'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'accuracy':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def _eval_accuracy(self, y_true, y_pred):
        '''
            compute Accuracy score averaged across tasks
        '''
        acc_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            acc = accuracy_score(y_true[is_labeled], y_pred[is_labeled])
            acc_list.append(acc)

        return {'accuracy': sum(acc_list) / len(acc_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_accuracy(y_true, y_pred)
