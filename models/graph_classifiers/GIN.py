import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from models.graph_classifiers.self_attention import SelfAttention


class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = [config['hidden_units'][0]] + config['hidden_units']
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.ga_heads = config['ga_heads']
        self.ga_recursive = config['ga_recursive']

        train_eps = config['train_eps']
        if config['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2
                
                self.linears.append(Linear(out_emb_dim, dim_target))
            

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input
        if self.ga_heads > 0:
            print('Creating GIN model with {} GA heads'.format(self.ga_heads))
            if self.ga_recursive:
                print('Applying GA recursively every layer')
            else:
                print('Applying GA every layer, but not recursively')
            self.selfatt = torch.nn.ModuleList([SelfAttention(num_heads=self.ga_heads, model_dim=out_emb_dim,
                                             dropout_keep_prob=1 - self.dropout) for _ in range(self.no_layers)])
            

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)

                if self.ga_heads > 0:
                    dense_x, valid_mask = torch_geometric.utils.to_dense_batch(x, batch=batch, fill_value=-1)
                    dense_x = self.selfatt[layer](dense_x, attn_mask=valid_mask.float())
                    possibly_attended_x = torch.masked_select(dense_x, torch.unsqueeze(valid_mask, -1)).reshape(x.shape)
                    if self.ga_recursive:
                        x = possibly_attended_x
                else:
                    possibly_attended_x = x
                out += F.dropout(self.pooling(self.linears[layer](possibly_attended_x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                if self.ga_heads > 0:
                    dense_x, valid_mask = torch_geometric.utils.to_dense_batch(x, batch=batch, fill_value=-1)
                    dense_x = self.selfatt[layer](dense_x, attn_mask=valid_mask.float())
                    possibly_attended_x = torch.masked_select(dense_x, torch.unsqueeze(valid_mask, -1)).reshape(x.shape)
                    if self.ga_recursive:
                        x = possibly_attended_x
                else:
                    possibly_attended_x = x
                out += F.dropout(self.linears[layer](self.pooling(possibly_attended_x, batch)), p=self.dropout, training=self.training)

        return out