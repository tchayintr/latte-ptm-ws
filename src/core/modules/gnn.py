import copy
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data
from torch_geometric.data import Batch


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_batch_clone(batch: Batch) -> Batch:
    return Batch.from_data_list(batch.to_data_list())


class GraphConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int,
                 gnn_type: str = 'gcn',
                 add_self_loops: bool = True,
                 bidirectional: bool = True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.gnn_type = gnn_type
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

        output_size = out_channels
        if gnn_type == 'gcn':
            graph_conv = GCNConv(in_channels,
                                 out_channels,
                                 add_self_loops=add_self_loops)
        elif gnn_type == 'gat':
            graph_conv = GATConv(in_channels,
                                 out_channels,
                                 heads=num_heads,
                                 add_self_loops=add_self_loops)
            output_size = out_channels * num_heads
        elif gnn_type == 'gtn':
            graph_conv = TransformerConv(in_channels,
                                         out_channels,
                                         heads=num_heads)
            output_size = out_channels * num_heads
        else:
            raise ValueError('invalid --gnn-type')

        if bidirectional:
            # clone 2 instances for forward and backward
            graph_conv = _get_clones(graph_conv, 2)
            output_size = output_size * 2

        self.graph_conv = graph_conv

        # reduce the dimension for calculating the residual connection
        self.W = nn.Linear(output_size, out_channels)


    def forward(self, x, edges):
        if self.bidirectional:
            forward_edges = edges[:, edges[0] < edges[1]]
            backward_edges = edges[:, edges[0] > edges[1]]
            edges_list = [forward_edges, backward_edges]

            ys = []
            for conv, edges in zip(self.graph_conv, edges_list):
                ys.append(conv(x, edges))

            y = torch.cat(ys, dim=-1)
        else:
            y = self.graph_conv(x, edges)

        return self.W(y)


class LatticeEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 num_heads: int,
                 dropout: float,
                 gnn_type: str = 'gcn',
                 add_self_loops: bool = True,
                 bidirectional: bool = True):
        super(LatticeEncoderLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.gnn_type = gnn_type
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

        self.gnn = GraphConv(d_model, d_model, num_heads, gnn_type,
                             add_self_loops, bidirectional)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x, edges):
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        x2 = self.gnn(x, edges)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x


class ShallowLatticeEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 num_heads: int,
                 dropout: float,
                 gnn_type: str = 'gcn',
                 add_self_loops: bool = True,
                 bidirectional: bool = True):
        super(ShallowLatticeEncoderLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.gnn_type = gnn_type
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

        self.gnn = GraphConv(d_model, dim_feedforward, num_heads, gnn_type,
                             add_self_loops, bidirectional)
        self.linear = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edges):
        x = self.gnn(x, edges)
        x = self.dropout(self.activation(self.linear(x)))
        x = self.norm(x)
        return x


class LatticeEncoder(nn.Module):
    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 gnn_type: str = 'gcn',
                 add_self_loops: bool = True,
                 bidirectional: bool = True,
                 shallow: bool = False):
        super(LatticeEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional
        self.shallow = shallow

        if self.shallow:
            encoder_layer = ShallowLatticeEncoderLayer(
                d_model=embed_size,
                dim_feedforward=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                gnn_type=gnn_type,
                add_self_loops=add_self_loops,
                bidirectional=bidirectional)
        else:
            encoder_layer = LatticeEncoderLayer(d_model=embed_size,
                                                dim_feedforward=hidden_size,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                gnn_type=gnn_type,
                                                add_self_loops=add_self_loops,
                                                bidirectional=bidirectional)

        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, batch: Batch) -> Batch:
        '''
        batch: lattice batch (torch_geometric.data.Batch)
        batch.x: embeddings of word indices in lattice (geo_batch.x)
        '''
        lat = _get_batch_clone(batch)
        x = lat.x
        num_total_nodes, embed_size = x.size()
        assert embed_size == self.embed_size

        for layer in self.layers:
            x = layer(x, lat.edge_index)
        lat.x = x
        return lat


if __name__ == '__main__':
    embed_size = 300
    hidden_size = 1024
    num_layers = 2
    num_heads = 4
    dropout = 0.2
    gnn_type = 'gcn'
    add_self_loops = True
    bidirectional = True
    shallow = False

    gnn = LatticeEncoder(embed_size,
                         hidden_size,
                         num_heads,
                         num_layers,
                         dropout,
                         gnn_type,
                         add_self_loops=add_self_loops,
                         bidirectional=bidirectional,
                         shallow=shallow)

    emb = nn.Embedding(100, 300)
    x = [
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3]),
    ]
    edges = [
        torch.tensor([[0, 1, 2, 3, 3], [1, 2, 4, 3, 0]]),
        torch.tensor([[0, 1, 2, 2], [1, 0, 1, 1]]),
    ]

    geo_batch = Batch.from_data_list([
        Data(x[0], edges[0]),
        Data(x[1], edges[1]),
    ])

    _geo_batch = _get_batch_clone(geo_batch)

    geo_batch.x = emb(_geo_batch.x)
    print(geo_batch.x)
    print(geo_batch.x.size())

    geo_batch_e = gnn(geo_batch)
    print(geo_batch_e.x)
    print(geo_batch_e.x.size())
