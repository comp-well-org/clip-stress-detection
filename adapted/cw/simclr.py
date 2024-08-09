import torch
import torch.nn as nn
import torch.nn.functional as F
from .auto_aug import AutoAUG
from .augmentations import jitter, scaling
from .nets import ProjectionHead, SigEncoder

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class ContrastiveLoss(nn.Module):
    """Contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.5,
    ):
        """Initialize the contrastive loss object.
        
        Args:
            temperature: The temperature for the contrastive loss.
        """
        super().__init__()
        self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            emb_i: The first embeddings of shape (batch_size, emb_dim).
            emb_j: The second embeddings of shape (batch_size, emb_dim).
        
        Returns:
            The contrastive loss.
        """
        batch_size, emb_dim = emb_i.shape
        assert emb_i.shape == emb_j.shape, 'Embeddings must have the same shape'
        assert batch_size > 1, 'Batch size must be greater than 1'
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2,
        )
        
        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones(
                (2 * batch_size,),
            ).to(emb_i.device).scatter_(
                0, torch.tensor([i]).to(emb_i.device), 0.,
            )
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature),
            )    
            loss_ij = -torch.log(numerator / denominator)
            return loss_ij.squeeze(0)
        
        loss = 0.
        for k in range(batch_size):
            loss += l_ij(k, k + batch_size) + l_ij(k + batch_size, k)
        return 1.0 / (2 * batch_size) * loss

class SimCLR(nn.Module):
    def __init__(
        self, 
        in_channel, 
        seq_encoder, 
        seq_emb_dim=128,
        proj_dim=128,
        use_leaves=False,
    ):
        super().__init__()
        self.use_leaves = use_leaves
        if use_leaves:
            self.view = AutoAUG(num_channel=in_channel)
        self.seq_encoder = SigEncoder(seq_encoder, seq_emb_dim)
        self.seq_proj = ProjectionHead(
            embedding_dim=seq_emb_dim,
            projection_dim=proj_dim,
        )
        self.loss_fn = ContrastiveLoss()
    
    def forward(self, common, tabular=None):
        if self.use_leaves:
            x1 = self.view(common)
            x2 = self.view(common)
        else:
            x1 = jitter(scaling(common, 0.03), 0.03)
            x2 = jitter(scaling(common, 0.03), 0.03)
        
        view1_emb = self.seq_encoder(x1)
        view2_emb = self.seq_encoder(x2)
        
        view1_emb = self.seq_proj(view1_emb)
        view2_emb = self.seq_proj(view2_emb)
        
        contrastive_loss = self.loss_fn(view1_emb, view2_emb)
        
        return contrastive_loss
