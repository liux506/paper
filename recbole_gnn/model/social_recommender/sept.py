import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix, eye
from torch_geometric.utils import degree

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import dgl.function as fn

from recbole_gnn.model.abstract_recommender import SocialRecommender
from recbole_gnn.model.layers import LightGCNConv

import dgl
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole_gnn.model.social_recommender.layer.cross_attention_layer import CrossTransformerEncoder


# from recbole_gnn.model.social_recommender.layer.gat_layer import GATLayer, CustomGATLayerEdgeReprFeat


class GatedGCNLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        # g.update_all(self.message_func,self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution
        '''
        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization
        '''
        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
        '''
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        '''
        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)


class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()
        '''
        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.UUR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.UVR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.USR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])'''

        self.UUR_layers = nn.ModuleList(
            [CrossTransformerEncoder(d_model, nhead, attention_type) for _ in range(layer_nums)])
        self.UVR_layers = nn.ModuleList(
            [CrossTransformerEncoder(d_model, nhead, attention_type) for _ in range(layer_nums)])
        self.USR_layers = nn.ModuleList(
            [CrossTransformerEncoder(d_model, nhead, attention_type) for _ in range(layer_nums)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, s_u, g, s_g):
        h = g.ndata['emb_h']
        N, D = s_u.shape
        s = h[:N]

        for layer in self.UUR_layers:
            x = layer(s, s_u)

        h_n = h.clone()
        h_n[:N] = x
        g.ndata['emb'] = h_n
        g.apply_edges(lambda edges: {'src1': edges.src['emb']})
        src = g.edata['src1']
        g.apply_edges(lambda edges: {'dst1': edges.dst['emb']})
        dst = g.edata['dst1']

        for layer in self.UVR_layers:
            edge = layer(src, dst)

        s_g.ndata['emb1'] = x
        s_g.ndata['emb'] = s
        s_g.apply_edges(lambda edges: {'src1': edges.src['emb']})
        src1 = s_g.edata['src1']
        s_g.apply_edges(lambda edges: {'dst1': edges.dst['emb1']})
        dst1 = s_g.edata['dst1']

        for layer in self.USR_layers:
            social_edge = layer(src1, dst1)

        return edge, social_edge  # [N, N, D]


class MERG(nn.Module):

    def __init__(self, in_dim, hidden_dim, max_node_num, global_layer_num=2, dropout=0.1):
        super().__init__()
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)

        self.edge_proj = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.edge_proj2 = nn.Linear(in_dim, hidden_dim)  # baseline4
        self.edge_proj3 = nn.Linear(in_dim, hidden_dim)
        self.edge_proj4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim  # baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)
        self.max_node_num = max_node_num
        self.CrossT = CrossTransformer(hidden_dim, nhead=2, layer_nums=1, attention_type='linear')

    def forward(self, g, h, e, s_u, s_g):
        g.apply_edges(lambda edges: {'src': edges.src['emb_h']})
        src = g.edata['src'].unsqueeze(1)  # [M,1,D]
        g.apply_edges(lambda edges: {'dst': edges.dst['emb_h']})
        dst = g.edata['dst'].unsqueeze(1)  # [M,1,D]
        edge = torch.cat((src, dst), 1).to(h.device)  # [M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)  # [M,D]
        lr_e_local = self.edge_proj2(lr_e_local)

        edge, sedge = self.CrossT(s_u, g, s_g)
        edge = edge.squeeze(0)
        sedge = sedge.squeeze(0)

        lr_e_global = self.edge_proj4(edge)
        sedge = self.edge_proj3(sedge)

        lr_e = lr_e_local + lr_e_global

        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)

        return lr_e, sedge

class MERG1(nn.Module):

    def __init__(self, in_dim, hidden_dim, max_node_num, global_layer_num=2, dropout=0.1):
        super().__init__()
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)

        self.edge_proj = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.edge_proj2 = nn.Linear(in_dim, hidden_dim)  # baseline4
        self.edge_proj3 = nn.Linear(in_dim, hidden_dim)
        self.edge_proj4 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim  # baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)
        self.max_node_num = max_node_num
        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')

    def forward(self, g, h, e, s_u, s_g):
        g.apply_edges(lambda edges: {'src': edges.src['emb_h']})
        src = g.edata['src'].unsqueeze(1)  # [M,1,D]
        g.apply_edges(lambda edges: {'dst': edges.dst['emb_h']})
        dst = g.edata['dst'].unsqueeze(1)  # [M,1,D]
        edge = torch.cat((src, dst), 1).to(h.device)  # [M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)  # [M,D]
        lr_e_local = self.edge_proj2(lr_e_local)

        lr_e = lr_e_local
        s_g.apply_edges(lambda edges: {'src': edges.src['emb_h']})
        src = s_g.edata['src'].unsqueeze(1)  # [M,1,D]
        s_g.apply_edges(lambda edges: {'dst': edges.dst['emb_h']})
        dst = s_g.edata['dst'].unsqueeze(1)  # [M,1,D]
        sedge = torch.cat((src, dst), 1).to(h.device)  # [M,2,D]
        sedge = self.edge_proj(sedge).squeeze(1)  # [M,D]
        sedge = self.edge_proj2(sedge)

        '''
        edge, sedge = self.CrossT(s_u, g, s_g)
        edge = edge.squeeze(0)
        sedge = sedge.squeeze(0)

        lr_e_global = self.edge_proj4(edge)
        sedge = self.edge_proj3(sedge)

        lr_e = lr_e_local + lr_e_global

        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)'''

        return lr_e, sedge



class SEPT(SocialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SEPT, self).__init__(config, dataset)

        # load dataset info
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        """Get sparse matrix that describe interactions between user_id and item_id."""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        """bulid sparse martix"""
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        self.g = dgl.from_scipy(A, device=self.device)
        self.g = dgl.add_self_loop(self.g)

        self.social_mat = dataset.net_matrix() + eye(self.n_users)
        self.s_g = dgl.from_scipy(self.social_mat, device=self.device)

        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        self._src_user = dataset.net_feat[dataset.net_src_field]
        self._tgt_user = dataset.net_feat[dataset.net_tgt_field]

        # load parameters info
        self.latent_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.drop_ratio = config["drop_ratio"]
        self.instance_cnt = config["instance_cnt"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.ssl_weight1 = config["ssl_weight1"]
        self.ssl_tau = config["ssl_tau"]
        self.ssl_tau1 = config["ssl_tau1"]
        self.require_pow = config['require_pow']
        dropout = config["dropout"]
        self.num_heads = config["num_heads"]
        self.batch_norm = config['batch_norm']
        self.residual = config['residual']
        self.edge_lr = config['edge_lr']

        # define layer and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.social_edge_index, self.social_edge_weight, self.sharing_edge_index, \
        self.sharing_edge_weight = self.get_user_view_matrix(dataset)

        # storage variables for full sort evaluation acceleration
        self.user_all_embeddings = None
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.layers = torch.nn.ModuleList()

        self.layers.append(
            GatedGCNLayer(self.latent_dim, self.latent_dim, dropout, self.batch_norm, self.residual))

        # self.layers.append(
        #   GATLayer(self.latent_dim, self.latent_dim, self.num_heads, dropout, self.batch_norm, self.residual))

        # self.layers = nn.ModuleList([CustomGATLayerEdgeReprFeat(self.latent_dim * self.num_heads, self.latent_dim, self.num_heads,
        # dropout, self.batch_norm, self.residual) for _ in range(1)])
        # self.layers.append(CustomGATLayerEdgeReprFeat(self.latent_dim * self.num_heads, self.latent_dim, 1, dropout, self.batch_norm, self.residual))

        self.merg = MERG(self.latent_dim, self.latent_dim, self.g.num_nodes())
        self.h = torch.nn.Embedding(num_embeddings=self.g.num_nodes(), embedding_dim=self.latent_dim)
        self.e = torch.nn.Embedding(num_embeddings=self.g.num_edges(), embedding_dim=self.latent_dim)
        self.embedding_h = nn.Linear(self.latent_dim, self.latent_dim * self.num_heads)
        self.embedding_e = nn.Linear(self.latent_dim, self.latent_dim * self.num_heads)

    def get_norm_edge_weight(self, edge_index, node_num):
        r"""Get normalized edge weight using the laplace matrix.
        """
        deg = degree(edge_index[0], node_num)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]
        return edge_weight

    def get_user_view_matrix(self, dataset):
        # Friend View: A_f = (SS) ⊙ S
        social_mat = dataset.net_matrix()
        social_matrix = social_mat.dot(social_mat)
        social_matrix = social_matrix.toarray() * social_mat.toarray() + eye(self.n_users)
        social_matrix = coo_matrix(social_matrix)
        social_edge_index = torch.stack([torch.LongTensor(social_matrix.row), torch.LongTensor(social_matrix.col)])
        social_edge_weight = self.get_norm_edge_weight(social_edge_index, self.n_users)

        # Sharing View: A_s = (RR^T) ⊙ S
        rating_mat = dataset.inter_matrix()
        sharing_matrix = rating_mat.dot(rating_mat.T)
        sharing_matrix = sharing_matrix.toarray() * social_mat.toarray() + eye(self.n_users)
        sharing_matrix = coo_matrix(sharing_matrix)
        sharing_edge_index = torch.stack([torch.LongTensor(sharing_matrix.row), torch.LongTensor(sharing_matrix.col)])
        sharing_edge_weight = self.get_norm_edge_weight(sharing_edge_index, self.n_users)

        return social_edge_index.to(self.device), social_edge_weight.to(self.device), \
               sharing_edge_index.to(self.device), sharing_edge_weight.to(self.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def subgraph_construction(self):
        r"""Perturb the joint graph to construct subgraph for integrated self-supervision signals.
        """

        def rand_sample(high, size=None, replace=True):
            return np.random.choice(np.arange(high), size=size, replace=replace)

        # perturb the raw graph with edge dropout
        keep = rand_sample(len(self._user), size=int(len(self._user) * (1 - self.drop_ratio)), replace=False)
        row = self._user[keep]
        col = self._item[keep] + self.n_users

        # perturb the social graph with edge dropout
        net_keep = rand_sample(len(self._src_user), size=int(len(self._src_user) * (1 - self.drop_ratio)),
                               replace=False)
        net_row = self._src_user[net_keep]
        net_col = self._tgt_user[net_keep]

        # concatenation and normalization
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index3 = torch.stack([net_row, net_col])
        edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        edge_weight = self.get_norm_edge_weight(edge_index, self.n_users + self.n_items)

        self.sub_graph = edge_index.to(self.device), edge_weight.to(self.device)

    def forward(self, graph=None):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]
        embeddingsS_list = [self.user_embedding.weight]

        if graph is None:  # for the original graph
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:  # for the augmented graph
            edge_index, edge_weight = graph

        all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
        norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        embeddings_list.append(norm_embeddings) #   4 3 1'''

        social_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        s_embeddings = self.gcn_conv(social_embeddings, self.social_edge_index, self.social_edge_weight)

        s_embeddings = F.normalize(s_embeddings, p=2, dim=1)

        e = self.e.weight
        self.g.ndata['emb_h'] = all_embeddings
        self.s_g.ndata['emb_h'] = s_embeddings
        e, s_e = self.merg(self.g, all_embeddings, e, s_embeddings, self.s_g)

        # for _ in range(self.n_layers):
        for gnn in self.layers:
            # all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
            all_embeddings, edge_embeddings = gnn(self.g, all_embeddings, e)
            # all_embeddings = gnn(self.g, all_embeddings)
            norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)

            social_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            s_embeddings = self.gcn_conv(social_embeddings, self.social_edge_index, self.social_edge_weight)
            s_embeddings = F.normalize(s_embeddings, p=2, dim=1)
            embeddingsS_list.append(s_embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        social_embeddings = torch.stack(embeddingsS_list, dim=1)
        social_embeddings = torch.sum(social_embeddings, dim=1)
        return user_all_embeddings, item_all_embeddings, social_embeddings

    def user_view_forward(self):
        all_social_embeddings = self.user_embedding.weight
        all_sharing_embeddings = self.user_embedding.weight
        social_embeddings_list = [all_social_embeddings]
        sharing_embeddings_list = [all_sharing_embeddings]

        for _ in range(self.n_layers):
            # friend view
            all_social_embeddings = self.gcn_conv(all_social_embeddings, self.social_edge_index, self.social_edge_weight)
            norm_social_embeddings = F.normalize(all_social_embeddings, p=2, dim=1)
            social_embeddings_list.append(norm_social_embeddings)
            # sharing view
            all_sharing_embeddings = self.gcn_conv(all_sharing_embeddings, self.sharing_edge_index, self.sharing_edge_weight)
            norm_sharing_embeddings = F.normalize(all_sharing_embeddings, p=2, dim=1)
            sharing_embeddings_list.append(norm_sharing_embeddings)

        social_all_embeddings = torch.stack(social_embeddings_list, dim=1)
        social_all_embeddings = torch.sum(social_all_embeddings, dim=1)

        sharing_all_embeddings = torch.stack(sharing_embeddings_list, dim=1)
        sharing_all_embeddings = torch.sum(sharing_all_embeddings, dim=1)

        return social_all_embeddings, sharing_all_embeddings

    def forward1(self, graph=None):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]

        if graph is None:  # for the original graph
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:  # for the augmented graph
            edge_index, edge_weight = graph

        for _ in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
            norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return user_all_embeddings, item_all_embeddings

    def label_prediction(self, emb, aug_emb):
        prob = torch.matmul(emb, aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1)
        return prob

    def sampling(self, logits):
        return torch.topk(logits, k=self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples

    def calculate_ssl_loss(self, aug_emb, positive, emb):
        pos_emb = aug_emb[positive]
        pos_score = torch.sum(emb.unsqueeze(dim=1).repeat(1, self.instance_cnt, 1) * pos_emb, dim=2)
        ttl_score = torch.matmul(emb, aug_emb.transpose(0, 1))
        pos_score = torch.sum(torch.exp(pos_score / self.ssl_tau), dim=1)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_tau), dim=1)
        ssl_loss = - torch.sum(torch.log(pos_score / ttl_score))
        return ssl_loss

    def ssl_loss(self, aug, pos, emb):
        u_emd1 = F.normalize(emb[pos], dim=1)
        u_emd2 = F.normalize(aug[pos], dim=1)
        all_user2 = F.normalize(aug, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl = -torch.sum(torch.log(v1 / v2))
        return ssl

    def calculate_rec_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        self.user_all_embeddings, item_all_embeddings, social_embeddings = self.forward()
        user1, item1 = self.forward1()
        u_embeddings = self.user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        ssl_user = self.ssl_loss(social_embeddings, user, self.user_all_embeddings)
        loss1 = self.ssl_weight1 * ssl_user
        loss += self.ssl_weight1 * ssl_user

        ssl_u = self.ssl_loss(user1, user, self.user_all_embeddings)
        loss2 = self.ssl_weight * ssl_u
        loss += self.ssl_weight * ssl_u

        '''
        ssl_i = self.ssl_loss(item1, pos_item, item_all_embeddings)
        loss3 = self.ssl_weight * ssl_i
        loss += self.ssl_weight * ssl_i
        '''

        return loss

    def calculate_loss(self, interaction):
        # preference view
        rec_loss = self.calculate_rec_loss(interaction)

        # L = L_r + β * L_{ssl}
        loss = rec_loss  #+ self.ssl_weight * ssl_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings, social_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)