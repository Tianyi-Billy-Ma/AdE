import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv, MessagePassing
from .layers import MLP
from torch.nn import Linear
import torch

class GIN(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 dropout
                 ):
        super(GIN, self).__init__()

        self.conv1 = GINConv(MLP(in_dim, hid_dim, hid_dim, num_layers=1), learn_eps=False)
        self.conv2 = GINConv(MLP(hid_dim, hid_dim, hid_dim, num_layers=1), learn_eps=False)
        self.lin1 = Linear(2 * hid_dim + in_dim, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, X, edge_index):

        edge_index = edge_index.coalesce()
        h1 = self.conv1(X, edge_index)
        h1 = F.relu(h1)

        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)

        h = torch.cat((X, h1, h2), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h
