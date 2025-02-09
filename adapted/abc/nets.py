import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.5,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class SigEncoder(nn.Module):
    def __init__(self, seq_encoder, seq_emb_dim=128):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            seq_encoder,
            nn.LazyLinear(seq_emb_dim),
        )
    
    def forward(self, common, tabular=None):
        seq_emb = self.seq_encoder(common)
        return seq_emb
