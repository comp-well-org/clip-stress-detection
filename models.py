import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import BertTokenizer, BertModel
from adapted.resnet1d import ResNet1D

warnings.filterwarnings('ignore')

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()

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

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

class CNNSeqEncoder(nn.Module):
    def __init__(
        self, in_channels, hidden_size_half, num_layers, dropout=0.1, kernel_size=3, padding=1,
    ):
        super().__init__()
        uplayers = nn.ModuleList()
        for _ in range(num_layers):
            uplayers.append(nn.Conv1d(
                in_channels, hidden_size_half, kernel_size=kernel_size, padding=padding,
            ))
            uplayers.append(nn.ReLU())
            uplayers.append(nn.Dropout(dropout))
            uplayers.append(nn.BatchNorm1d(hidden_size_half))
            in_channels = hidden_size_half
            hidden_size_half *= 2
        self.uplayers = nn.Sequential(*uplayers)
        downlayers = nn.ModuleList()
        for _ in range(num_layers):
            downlayers.append(nn.Conv1d(
                in_channels, hidden_size_half, kernel_size=kernel_size, padding=padding,
            ))
            downlayers.append(nn.ReLU())
            downlayers.append(nn.Dropout(dropout))
            downlayers.append(nn.BatchNorm1d(hidden_size_half))
            in_channels = hidden_size_half
            hidden_size_half //= 2
        self.downlayers = nn.Sequential(*downlayers)
    
    def forward(self, x):
        x = self.uplayers(x)
        x = self.downlayers(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        x = x.squeeze()
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, dropout):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_size, nhead, hidden_size, dropout, batch_first=True), 
            num_layers,
        )

    def forward(self, x):
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = nn.functional.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        return x

class ResNetSeqEncoder(nn.Module):
    def __init__(
        self, in_channels, base_filters, hidden_size, 
        n_block=4, kernel_size=3,
    ):
        super().__init__()
        self.tab_resnet = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=1,
            groups=1,
            n_block=n_block,
            n_classes=hidden_size,
        )
        
    def forward(self, x):
        return self.tab_resnet(x)

class BertEncoder(nn.Module):
    def __init__(self, name='bert-base-uncased', device='cpu'):
        super().__init__()
        self.model = BertModel.from_pretrained(name)
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def forward(self, x):
        encoding = self.tokenizer.batch_encode_plus(
            x,	 
            padding=True,			
            truncation=True,	
            return_tensors='pt',
            add_special_tokens=True,
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = out.last_hidden_state
            sentence_embedding = word_embeddings.mean(dim=1) 
        return sentence_embedding

class MultiModalEncoder(nn.Module):
    def __init__(
        self, fitbit_net, tab_net, dim=32,
        n_classes=None,
    ):
        super().__init__()
        # encoders for each modality
        self.fitbit_net = fitbit_net
        self.tab_net = tab_net
        
        availability = {}
        availability['fitbit'] = fitbit_net is not None
        availability['tab'] = tab_net is not None
        self.availability = availability
        self.fc = nn.LazyLinear(dim)
        if n_classes:
            self.fc2 = nn.LazyLinear(n_classes)
        else:
            self.fc2 = None
    
    def load_pretrained(self, pretrained):
        self.fitbit_net.load_state_dict(pretrained.state_dict())

    def forward(self, common, tabular):
        rep = []
        
        # common fitbit modalities
        common_enc = self.fitbit_net(common)
        if len(common_enc.shape) == 1:
            common_enc = common_enc.unsqueeze(0)
        
        # tabular modalities
        tab_enc = self.tab_net(tabular)
        if len(tab_enc.shape) == 1:
            tab_enc = tab_enc.unsqueeze(0)
        
        # concatenate the representations
        rep.append(common_enc)
        rep.append(tab_enc)
        ans = torch.cat(rep, dim=1)
        
        # linear projection
        ans = self.fc(ans)
        
        # if n_classes is not None, apply another linear layer
        if self.fc2:
            # activation function
            ans = nn.functional.relu(ans)
            ans = self.fc2(ans)
        
        return ans

class CLIP(nn.Module):
    def __init__(
        self,
        text_encoder,
        seq_net, tab_net, 
        seq_emb_dim=128,
        text_emb_dim=768,
        proj_dim=128,
        temperature=0.07,
    ):
        super().__init__()
        self.seq_encoder = MultiModalEncoder(
            fitbit_net=seq_net,
            tab_net=tab_net,
            dim=seq_emb_dim,
        )
        self.text_encoder = text_encoder
        self.seq_proj = ProjectionHead(
            embedding_dim=seq_emb_dim,
            projection_dim=proj_dim,
        )
        self.text_proj = ProjectionHead(
            embedding_dim=text_emb_dim,
            projection_dim=proj_dim,
        )
        self.temperature = temperature
    
    def forward(self, texts, common, tabular):
        # embeddings for sequences and texts
        seq_features = self.seq_encoder(common, tabular)
        seq_embeddings = self.seq_proj(seq_features)
        text_embeddings = self.text_proj(texts)
        
        # compute the loss
        logits = (text_embeddings @ seq_embeddings.T) / self.temperature
        seq_similarity = seq_embeddings @ seq_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (seq_similarity + text_similarity) / 2 * self.temperature, dim=-1,
        )
        txt_loss = cross_entropy(logits, targets, reduction='none')
        seq_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (txt_loss + seq_loss) / 2
        return loss.mean()

class LinearProbe(nn.Module):
    def __init__(self, model, out_dim, freeze=True):
        super().__init__()
        self.seq_encoder = copy.deepcopy(model.seq_encoder)
        self.seq_proj = copy.deepcopy(model.seq_proj)
        self.linear = torch.nn.LazyLinear(out_dim)
        
        # freeze the seq_encoder and seq_proj
        if freeze:
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
            for param in self.seq_proj.parameters():
                param.requires_grad = False

    def forward(self, common, tabular):
        x = self.seq_encoder(common, tabular)
        x = self.seq_proj(x)
        x = nn.functional.relu(x)
        x = self.linear(x)
        return x

class SignalSSL(nn.Module):
    def __init__(self, model, out_dim, freeze=True):
        super().__init__()
        self.seq_encoder = copy.deepcopy(model.seq_encoder)
        self.seq_proj = copy.deepcopy(model.seq_proj)
        self.linear = torch.nn.LazyLinear(out_dim)
        
        # freeze the seq_encoder and seq_proj
        if freeze:
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
            for param in self.seq_proj.parameters():
                param.requires_grad = False

    def forward(self, common):
        x = self.seq_encoder(common)
        x = self.seq_proj(x)
        x = nn.functional.relu(x)
        x = self.linear(x)
        return x

class SupBaselineNet(nn.Module):
    def __init__(self, seq_net, tab_net, seq_emb_dim=128, proj_dim=128, out_dim=2):
        super().__init__()
        self.seq_encoder = MultiModalEncoder(
            fitbit_net=seq_net,
            tab_net=tab_net,
            dim=seq_emb_dim,
        )
        self.seq_proj = ProjectionHead(
            embedding_dim=seq_emb_dim,
            projection_dim=proj_dim,
        )
        self.linear = torch.nn.LazyLinear(out_dim)
    
    def forward(self, common, tabular):
        x = self.seq_encoder(common, tabular)
        x = self.seq_proj(x)
        x = nn.functional.relu(x)
        x = self.linear(x)
        return x
