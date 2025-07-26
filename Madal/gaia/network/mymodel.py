import torch, math
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import sys
from network.cross_modal import TripleModalityLadderMerge, CrossMerge
from torch.autograd import Variable
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.utils import dense_to_sparse
from se_net import SE_Block
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
  
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, dropout = 0.1, dim_feedforward = 64, gru_hidden_size = 64, tf_layer = 2, nhead = 4):
        super(FeatureExtractor, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model= gru_hidden_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout= dropout,
            batch_first=True)
        
        self.gru = nn.GRU(
            input_size = input_dim,
            hidden_size = gru_hidden_size,
            num_layers = 1,
            batch_first = True,
            bidirectional= False
        )        

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = tf_layer)  

    def forward(self, x):
        o, _ = self.gru(x)
        o = self.transformer_encoder(o)
        return o 

class GatMetricEncoder(nn.Module):
    def __init__(self, input_dim, device, seq_len = 25, dropout = 0.1, dim_feedforward = 64, nhead = 4, gat_heads = 4, gat_iteration = 2, tf_layer = 2, out_channels = 16):
        super(GatMetricEncoder, self).__init__()
        self.input_dim = input_dim
        self.gat_iteration = gat_iteration
        self.seq_len = seq_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model= gat_heads * out_channels,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout= dropout,
            batch_first=True)
        
        self.stacked_gat = nn.ModuleList(
            [ GATConv(seq_len, out_channels, gat_heads, dropout=dropout) ]  + 
            [ GATConv(gat_heads * out_channels, out_channels, gat_heads, dropout=dropout) for _ in range(gat_iteration-1) ]
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = tf_layer) 
        self.device = device

    def forward(self, metrics):
        bs = metrics.shape[0]
        metrics = metrics.permute(0,2,1).reshape(-1, self.seq_len) 
        adj = torch.ones(self.input_dim, self.input_dim)  
        edge_index, _ = dense_to_sparse(adj)   
        edges = torch.concat([edge_index + self.input_dim * i for i in range(bs)],dim=1).cuda()

        for i in range(self.gat_iteration):
            metrics_new = self.stacked_gat[i](metrics,edges)
            if i>=1:
                metrics_new = (metrics + metrics_new) 
            metrics = metrics_new  

        metrics = metrics.reshape(bs, self.input_dim, -1)
        metrics = self.transformer_encoder(metrics)  
        return metrics




    def forward(self, log_seq: torch.Tensor, log_ts: torch.Tensor):
        
        *rest, _, _ = log_seq.shape
        log_seq = log_seq.reshape(-1, log_seq.shape[-2], log_seq.shape[-1]) 
        log_seq = self.pe(log_seq)

        log_seq = self.log_seq_encoder(log_seq)

        log_seq = log_seq.view(*rest, log_seq.shape[-2], log_seq.shape[-1])     
        
        log_seq = self.pool(log_seq).squeeze()   
        log_ts = self.log_ts_encoder(log_ts)
        log_feature = self.fuse_net(log_seq, log_ts) 

        return log_feature   



class LogGraphEncoder(nn.Module):
    def __init__(self, log_sta_dim, out_dim, n_head, log_seq_dim, seq_len, window_size):
        super(LogGraphEncoder, self).__init__()
        self.log_sta_encoder = FeatureExtractor(log_sta_dim,gru_hidden_size=out_dim, nhead=n_head)
        self.log_seq_encoder = FeatureExtractor(log_seq_dim, gru_hidden_size=out_dim, nhead=n_head)
        self.seq_len = seq_len
        self.ffn = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512,window_size),
            nn.LayerNorm(window_size),
            nn.ReLU()
        )
        self.cross_module = CrossMerge(out_dim)

    def forward(self, log_seq: torch.Tensor, log_ts: torch.Tensor):
        log_ts = self.log_sta_encoder(log_ts)

        log_seq = log_seq[:,:self.seq_len,:]
        log_seq = self.log_seq_encoder(log_seq)
        log_seq = log_seq.permute(0,2,1)
        batch_size, hidden_dim, seq_len = log_seq.shape 
        log_seq= log_seq.reshape(-1, seq_len)
        log_seq = self.ffn(log_seq)
        log_seq = log_seq.reshape(batch_size, hidden_dim,-1).permute(0, 2, 1)
        
        return self.cross_module(log_ts, log_seq)  #[bs, window_size, out_dim]


class AlignerNet(nn.Module):
    def __init__(self, shared_dim, seq_len):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        self.pool = nn.AvgPool1d(kernel_size=seq_len)


    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x = x.reshape(-1, hidden_dim) 
        x = self.aligner(x)            
        x = x.reshape(batch_size, seq_len, hidden_dim)  

        x = x.permute(0, 2, 1)  
        x = self.pool(x)       
       
        return x


class SimilarityModule(nn.Module):
    def __init__(self,  metric_feature_dim, trace_feature_dim, log_feature_dim, log_seq_dim,
                  device, input_dim = 64,gat_head = 4, metric_nhead = 4, trace_nhead = 4, log_nhead = 8,
                 gat_out_channel = 16, window_size = 25, log_seq_len = 2048):
        
        super(SimilarityModule,self).__init__()
        assert gat_out_channel * gat_head == input_dim, "dim unmatched"

        self.metric_feature_extractor = GatMetricEncoder(metric_feature_dim, device, nhead=metric_nhead, gat_heads= gat_head, out_channels=gat_out_channel, seq_len=window_size)
        self.trace_feature_extractor = FeatureExtractor(trace_feature_dim, gru_hidden_size = input_dim, nhead=trace_nhead)
        self.log_feature_extractor = LogGraphEncoder(log_sta_dim=log_feature_dim, out_dim = input_dim,log_seq_dim = log_seq_dim, n_head=log_nhead,window_size=window_size,seq_len=log_seq_len)
       
        self.metric_aligner = AlignerNet(input_dim, metric_feature_dim)
        self.trace_aligner = AlignerNet(input_dim, window_size)
        self.log_aligner = AlignerNet(input_dim, window_size)
    

    def forward(self, m, t, l, l_s):
        m = self.metric_feature_extractor(m)
        l = self.log_feature_extractor(l_s, l)   
        t = self.trace_feature_extractor(t)


        aligned_m = self.metric_aligner(m).squeeze(-1)    
        aligned_l = self.log_aligner(l).squeeze(-1)    
        aligned_t = self.trace_aligner(t).squeeze(-1)    

        return aligned_m, aligned_l, aligned_t


class MultiModalDetection(nn.Module):
    def __init__(self,  metric_feature_dim, trace_feature_dim, log_feature_dim, log_seq_dim,
                 device, input_dim = 64, gat_head = 4, metric_nhead = 4, trace_nhead = 4, log_nhead = 8,
                 gat_out_channel = 16,  window_size = 25, log_seq_len = 2048, param_center=True):
        super(MultiModalDetection, self).__init__()
        assert gat_out_channel * gat_head == input_dim, "dim unmatched"
        self.senet = SE_Block(4)

        self.metric_feature_extractor = GatMetricEncoder(metric_feature_dim, device, nhead=metric_nhead, gat_heads= gat_head, out_channels=gat_out_channel, seq_len=window_size)
        self.trace_feature_extractor = FeatureExtractor(trace_feature_dim, gru_hidden_size = input_dim, nhead=trace_nhead)
        self.log_feature_extractor = LogGraphEncoder(log_sta_dim=log_feature_dim, out_dim = input_dim,log_seq_dim = log_seq_dim, n_head=log_nhead,window_size=window_size,seq_len=log_seq_len)

        self.metric_mapping = AlignerNet(input_dim, metric_feature_dim)
        self.trace_mapping = AlignerNet(input_dim, window_size)
        self.log_mapping = AlignerNet(input_dim, window_size)

        self.fuse_net = TripleModalityLadderMerge(input_dim,input_dim,4) 
        self.param_center =  param_center

        if self.param_center:
            self.m_center = nn.Parameter(torch.ones(input_dim))
            self.l_center = nn.Parameter(torch.ones(input_dim))
            self.t_center = nn.Parameter(torch.ones(input_dim))
            self.fused_center = nn.Parameter(torch.ones(input_dim))

            nn.init.normal_(self.m_center)
            nn.init.normal_(self.l_center)
            nn.init.normal_(self.t_center)
            nn.init.normal_(self.fused_center)

    def forward(self, m, t, l, l_s, aligned_m, aligned_t, aligned_l):
        m = self.metric_feature_extractor(m)
        l = self.log_feature_extractor(l_s, l)  
        t = self.trace_feature_extractor(t)

        m = self.metric_mapping(m).unsqueeze(1)  
        l = self.log_mapping(l).unsqueeze(1) 
        t = self.trace_mapping(t).unsqueeze(1) 


        correlation = self.fuse_net(m.squeeze(-1), t.squeeze(-1), l.squeeze(-1)).squeeze()
        correlation = correlation.unsqueeze(1).unsqueeze(-1)
       
        attention_score = self.senet(torch.cat([m, l, t, correlation], 1))  
        
        cos_m_t = F.cosine_similarity(aligned_m, aligned_t, dim=1)  
        cos_m_l = F.cosine_similarity(aligned_m, aligned_l, dim=1)
        cos_t_l = F.cosine_similarity(aligned_t, aligned_l, dim=1)
        
        skl_m = 1.0 - (cos_m_t + cos_m_l) / 2
        skl_t = 1.0 - (cos_m_t + cos_t_l) / 2
        skl_l = 1.0 - (cos_t_l + cos_m_l) / 2
        skl_f = 1.0 - torch.maximum(torch.maximum(skl_m,skl_t),skl_l)
        skl_score = torch.concat([skl_m.unsqueeze(1), skl_l.unsqueeze(1), skl_t.unsqueeze(1), skl_f.unsqueeze(1)], dim=1)

        return attention_score.squeeze(), skl_score, m, l, t, correlation

class UniReconstructNet(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super(UniReconstructNet, self).__init__()
        self.recnet = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim * seq_len),
            nn.ReLU()
        )

        self.output_dim = output_dim
        self.seq_len = seq_len

    def forward(self, x):
        x = self.recnet(x)
        x = x.reshape(-1, self.seq_len, self.output_dim)
        return x

class MultiReconstructNet(nn.Module):
    def __init__(self, input_dim, window_size, metric_dim, log_embed_dim, log_event_num, trace_dim):
        super(MultiReconstructNet, self).__init__()
        self.m_rec = UniReconstructNet(input_dim, metric_dim, window_size)
        self.l_rec = UniReconstructNet(input_dim, log_event_num, window_size)
        self.t_rec = UniReconstructNet(input_dim, trace_dim, window_size)

        self.fused_m_rec = UniReconstructNet(input_dim, metric_dim, window_size)
        self.fused_l_rec = UniReconstructNet(input_dim, log_event_num, window_size)
        self.fused_t_rec = UniReconstructNet(input_dim, trace_dim, window_size)
       
        self.window_size = window_size
        self.metric_dim = metric_dim
        self.log_embed_dim = log_embed_dim
        self.log_event_num = log_event_num
        self.trace_dim = trace_dim


    def forward(self, x, m, t, l):
        rec_m, rec_t, rec_l = self.m_rec(m), self.t_rec(t), self.l_rec(l)

        fused_rec_m, fused_rec_t, fused_rec_l = \
                self.fused_m_rec(x), self.fused_t_rec(x), self.fused_l_rec(x)

        return rec_m, rec_t, rec_l, fused_rec_m, fused_rec_t, fused_rec_l