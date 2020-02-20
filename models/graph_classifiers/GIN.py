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
        self.ga_every_layer = config['ga_every_layer']

        train_eps = config['train_eps']

        # TOTAL NUMBER OF PARAMETERS #

        # first: dim_features*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target
        # l-th: input_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target

        # -------------------------- #

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                #self.linears.append(Linear(dim_features, dim_target))
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2
                
                self.linears.append(Linear(out_emb_dim, dim_target))
            

        #self.first_h = torch.nn.ModuleList(self.first_h)
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input
        if self.ga_heads > 0:
            print('Creating GIN model with {} GA heads'.format(self.ga_heads))
            if self.ga_every_layer:
                print('Performing GA every layer')
                self.selfatt = [SelfAttention(num_heads=self.ga_heads, model_dim=out_emb_dim,
                                             dropout_keep_prob=1 - self.dropout) for _ in range(self.no_layers)]
                self.ga_layers = [Linear(2 * out_emb_dim, out_emb_dim) for _ in range(self.no_layers)]
            else:
                self.selfatt = SelfAttention(num_heads=self.ga_heads, model_dim=out_emb_dim, dropout_keep_prob=1-self.dropout)
                self.selfatt_linear = Linear(out_emb_dim, dim_target)

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        if self.config.dataset.name in ["NCI1", "DD", "PROTEINS", "ENZYMES"]:
            pooling = global_add_pool 
        else:
            pooling = global_mean_pool

        for layer in range(self.no_layers):
            # print(f'Forward: layer {l}')
            if layer == 0:
                x = self.first_h(x)
                if self.ga_heads > 0 and self.ga_every_layer is True:
                    dense_x, valid_mask = torch_geometric.utils.to_dense_batch(x, batch=batch, fill_value=-1)
                    dense_x = self.selfatt[layer](dense_x, attn_mask=valid_mask.float())
                    x = torch.cat([x, torch.masked_select(dense_x, torch.unsqueeze(valid_mask, -1)).reshape(x.shape)], dim=-1)
                    x = torch.nn.functional.relu(self.ga_layers[layer](x))
                out += F.dropout(pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                if self.ga_heads > 0 and self.ga_every_layer is True:
                    dense_x, valid_mask = torch_geometric.utils.to_dense_batch(x, batch=batch, fill_value=-1)
                    dense_x = self.selfatt[layer](dense_x, attn_mask=valid_mask.float())
                    x = torch.cat([x, torch.masked_select(dense_x, torch.unsqueeze(valid_mask, -1)).reshape(x.shape)], dim=-1)
                    x = torch.nn.functional.relu(self.ga_layers[layer](x))
                out += F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)

        if self.ga_heads > 0 and self.ga_every_layer is False:
            dense_x, valid_mask = torch_geometric.utils.to_dense_batch(x, batch=batch, fill_value=-1)
            dense_x = self.selfatt(dense_x, attn_mask=valid_mask.float())
            gathered_x = torch.masked_select(dense_x, torch.unsqueeze(valid_mask, -1)).reshape(x.shape)
            out += F.dropout(self.selfatt_linear(pooling(gathered_x, batch)), p=self.dropout, training=self.training)
        return out