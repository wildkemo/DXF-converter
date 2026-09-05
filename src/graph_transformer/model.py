import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformer(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, num_layers=4, heads=4, dropout=0.1):
        super(GraphTransformer, self).__init__()
        
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            # TransformerConv supports edge features with edge_dim
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // heads, 
                    heads=heads, 
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        
        # Output MLP predicting the deltas
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # 1. Embed node and edge features with LayerNorm
        x = self.node_norm(F.relu(self.node_emb(x)))
        edge_attr = self.edge_norm(F.relu(self.edge_emb(edge_attr)))
        
        # 2. Graph Transformer Layers with pre-norm residual
        for conv, norm in zip(self.layers, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(F.relu(x))
            x = self.dropout(x)
            x = x + x_res  # Residual connection
            
        # 3. Coordinate Prediction Head
        out = self.head(x)
        return out
