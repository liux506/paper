import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_gnn.model.social_recommender.layer.cross_attention_layer import CrossTransformerEncoder

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()

        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.FAM_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.ARM_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, qfea, kfea, mask0=None, mask1=None):
        """
        Args:
            qfea (torch.Tensor): [B, N, D]
            kfea (torch.Tensor): [B, D]
            mask0 (torch.Tensor): [B, N] (optional)
            mask1 (torch.Tensor): [B, N] (optional)
        """

        B, N, D = qfea.shape
        kfea = kfea.unsqueeze(1).repeat(1, N, 1)  # [B,N,D]

        mask1 = torch.ones([B, N]).to(qfea.device)
        for layer in self.FAM_layers:
            qfea = layer(qfea, kfea, mask0, mask1)
            # kfea = layer(kfea, qfea, mask1, mask0)

        qfea_end = qfea.repeat(1, 1, N).view(B, -1, D)
        qfea_start = qfea.repeat(1, N, 1).view(B, -1, D)
        # mask2 = mask0.repeat([1,N])
        for layer in self.ARM_layers:
            # qfea_start = layer(qfea_start, qfea_end, mask2, mask2)
            qfea_start = layer(qfea_start, qfea_end)

        return qfea_start.view([B, N, N, D])  # [B,N*N,D]

class MERG(nn.Module):
    def __init__(self, in_dim, hidden_dim, global_layer_num=2, dropout=0.1):
        super().__init__()
        '''
        self.edge_proj = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.edge_proj2 = nn.Linear(in_dim, hidden_dim)  # baseline4
        self.edge_proj3 = nn.Linear(in_dim, hidden_dim)
        self.edge_proj4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim  # baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)

        self.global_layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          True, True) for _ in range(global_layer_num - 1)])
        self.global_layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout, True, True))

        # self.global_layers = nn.ModuleList([ ResidualAttentionBlock( d_model = hidden_dim, n_head = 1)
        #                                    for _ in range(global_layer_num) ])

        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')
        '''
        super().__init__()
        self.bn_node_lr_e_local = nn.BatchNorm1d(hidden_dim)
        self.bn_node_lr_e_global = nn.BatchNorm1d(hidden_dim)
        self.proj1 = nn.Linear(in_dim, hidden_dim ** 2)
        self.proj2 = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.edge_proj2 = nn.Linear(in_dim, hidden_dim)
        self.edge_proj3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        # self.bn_local = nn.BatchNorm1d(in_dim) #baseline4'
        self.bn_local = nn.LayerNorm(in_dim)
        self.bn_global = nn.BatchNorm1d(hidden_dim)  # baseline4

    def forward(self, g, h, e):
        g.apply_edges(lambda edges: {'src': edges.src['emb_h']})
        src = g.edata['src'].unsqueeze(1)  # [M,1,D]
        g.apply_edges(lambda edges: {'dst': edges.dst['emb_h']})
        dst = g.edata['dst'].unsqueeze(1)  # [M,1,D]
        edge = torch.cat((src, dst), 1).to(h.device)  # [M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)  # [M,D]
        lr_e_local = self.edge_proj2(lr_e_local)

        '''
        hs = g.ndata['emb_h']
        Ng = g.number_of_nodes()
        padding = nn.ConstantPad1d((0, g.number_of_nodes() - Ng), 0)
        pad_h = padding(hs.T).T  # [Nmax, D] #feat
        hs = pad_h.unsqueeze(0).to(h.device)  # [B,Nmax,Din]
        mask0 = torch.ones((1, g.number_of_nodes())).cuda()
        
        # Gated-GCN for extract global feature
        hs2g = h
        for conv in self.global_layers:
            hs2g, _ = conv(g, hs2g, e)
        g.ndata['hs2g'] = hs2g
        global_g = dgl.mean_nodes(g, 'hs2g')  # [B,D]

        edge = self.CrossT(hs, global_g, mask0).squeeze(0)  # [B,N,N,D]

        index_edge = edge[g.all_edges()[0], g.all_edges()[1], :]
        lr_e_global = self.edge_proj4(index_edge)

        lr_e = e + lr_e_local + lr_e_global

        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)
        '''


        '''
        N = h.shape[0] # 2027
        h_proj1 = F.dropout(F.relu(self.proj1(h)), 0.1, training=self.training) # [2627,4096]
        h_proj1 = h_proj1.view(-1, self.hidden_dim)
        h_proj2 = F.dropout(F.relu(self.proj2(h)), 0.1, training=self.training)
        h_proj2 = h_proj2.permute(1, 0)
        mm = torch.mm(h_proj1, h_proj2)
        mm = mm.view(N, self.hidden_dim, -1).permute(0, 2, 1)  # [N, N, D]
        lr_e_global = mm[g.all_edges()[0], g.all_edges()[1], :]  # [M,D]

        lr_e_global = self.edge_proj3(self.bn_global(lr_e_global))
        # bn=>relu=>dropout
        lr_e_global = self.bn_node_lr_e_global(lr_e_global)
        lr_e_global = F.relu(lr_e_global)
        lr_e_global = F.dropout(lr_e_global, 0.1, training=self.training)

        lr_e_local = self.bn_node_lr_e_local(lr_e_local)
        lr_e_local = F.relu(lr_e_local)
        lr_e_local = F.dropout(lr_e_local, 0.1, training=self.training)

        e = lr_e_local + lr_e_global + e  # baseline4
        '''

        return e


class CustomGATHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm, edge_lr=True):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.fc_h = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_e = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_proj = nn.Linear(3 * out_dim, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.batchnorm_e = nn.BatchNorm1d(out_dim)

        self.edge_lr = edge_lr
        if self.edge_lr:
            self.merg = MERG(in_dim, out_dim)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'], edges.dst['z_h']], dim=1)
        e_proj = self.fc_proj(z)
        attn = F.leaky_relu(self.attn_fc(z))
        return {'attn': attn, 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        # 19526 64 168116 64
        # g.ndata['local']  = h
        # if self.edge_lr:
        e = self.merg(g, h, e)

        z_h = self.fc_h(h)  # [2627, 32]
        z_e = self.fc_e(e)  # [161616, 32]
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e

        g.apply_edges(self.edge_attention)

        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        e = g.edata['e_proj']

        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)

        h = F.elu(h)
        e = F.elu(e)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e


class CustomGATLayerEdgeReprFeat(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True, edge_lr=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim * num_heads):
            self.residual = False

        self.heads = nn.ModuleList()

        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerEdgeReprFeat(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat'

        self.edge_lr = edge_lr
        #if self.edge_lr:
        self.merg = MERG(in_dim, in_dim)

    def forward(self, g, h, e):

        g.ndata['emb_h'] = h

        #e = self.merg(g, h, e)

        h_in = h  # for residual connection
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)

        if self.merge == 'cat':
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        else:
            raise NotImplementedError

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)


