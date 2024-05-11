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

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)

class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()

        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.HUR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.UVR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.UUR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, s_u, g, s_g):
        h = g.ndata['emb_h']
        N, D = s_u.shape
        s = h[:N]

        for layer in self.HUR_layers:
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

        s_g.ndata['emb'] = x
        s_g.apply_edges(lambda edges: {'src1': edges.src['emb']})
        src = s_g.edata['src1']
        s_g.apply_edges(lambda edges: {'dst1': edges.dst['emb']})
        dst = s_g.edata['dst1']

        for layer in self.UUR_layers:
            social_edge = layer(src, dst)

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
        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')

    def forward(self, g, h, e, s_u, s_g):

        g.apply_edges(lambda edges: {'src': edges.src['emb_h']})
        src = g.edata['src'].unsqueeze(1)  # [M,1,D]
        g.apply_edges(lambda edges: {'dst': edges.dst['emb_h']})
        dst = g.edata['dst'].unsqueeze(1)  # [M,1,D]
        edge = torch.cat((src, dst), 1).to(h.device)  # [M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)  # [M,D]
        lr_e_local = self.edge_proj2(lr_e_local)

        hs = g.ndata['emb_h']

        edge, social_edge = self.CrossT(s_u, g, s_g)
        edge.squeeze(0)
        social_edge.squeeze(0)

        lr_e_global = self.edge_proj4(edge)

        lr_e = lr_e_global + lr_e_local

        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)

        social_edge = self.bn_node_lr_e(social_edge)

        return lr_e, social_edge


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
        self.ssl_tau = config["ssl_tau"]
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

        self.soc_edge_index, self.soc_edge_weight = self.get_soc_matrix(dataset)

        # storage variables for full sort evaluation acceleration
        self.user_all_embeddings = None
        self.restore_user_e = None
        self.restore_item_e = None


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

        self.social_mat = dataset.net_matrix() #+ eye(self.n_users)


        self.s_g = dgl.from_scipy(self.social_mat, device=self.device)
        self.s_g = dgl.add_self_loop(self.s_g)
        self.s_e = torch.nn.Embedding(num_embeddings=self.s_g.num_edges(), embedding_dim=self.latent_dim)
        self.embeddings_e = nn.Linear(self.latent_dim, self.latent_dim * self.num_heads)

        #self.mergSocial = MERGSocial(self.latent_dim, self.latent_dim, self.s_g.num_nodes())

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_edge_weight(self, edge_index, node_num):
        r"""Get normalized edge weight using the laplace matrix.
        """
        deg = degree(edge_index[0], node_num)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]
        return edge_weight

    def get_soc_matrix(self, dataset):
        social_matrix = dataset.net_matrix() + eye(self.n_users)
        social_matrix = coo_matrix(social_matrix)
        soc_edge_index = torch.stack([torch.LongTensor(social_matrix.row), torch.LongTensor(social_matrix.col)])
        soc_edge_weight = self.get_norm_edge_weight(soc_edge_index, self.n_users)

        return soc_edge_index.to(self.device), soc_edge_weight.to(self.device)

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
        '''
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        '''
        ego_embeddings = self.h.weight
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

        if graph is None:  # for the original graph
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:  # for the augmented graph
            edge_index, edge_weight = graph

        all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
        norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        embeddings_list.append(norm_embeddings)

        social_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        s_embeddings = self.gcn_conv(social_embeddings, self.social_edge_index, self.social_edge_weight)
        s_embeddings = F.normalize(s_embeddings, p=2, dim=1)

        e = self.e.weight
        self.g.ndata['emb_h'] = all_embeddings
        e, s_e = self.merg(self.g, all_embeddings, e, s_embeddings, self.s_g)
        '''
        s_e = self.s_e.weight
        self.s_g.ndata['emb_h'] = s_embeddings
        s_e = self.mergSocial(self.s_g, social_embeddings, s_e, s_embeddings)
        '''
        # for _ in range(self.n_layers):
        for gnn in self.layers:
            # all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
            all_embeddings, edge_embeddings = gnn(self.g, all_embeddings, e)
            # all_embeddings = gnn(self.g, all_embeddings)
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

    def calculate_rec_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        self.user_all_embeddings, item_all_embeddings = self.forward()
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

        return loss

    def calculate_loss(self, interaction):
        # preference view
        rec_loss = self.calculate_rec_loss(interaction)
        '''
        # unlabeled sample view
        aug_user_embeddings, _ = self.forward(graph=self.sub_graph)

        # friend and sharing views
        friend_view_embeddings, sharing_view_embeddings = self.user_view_forward()

        user = interaction[self.USER_ID]
        aug_u_embeddings = aug_user_embeddings[user]
        social_u_embeddings = friend_view_embeddings[user]
        sharing_u_embeddings = sharing_view_embeddings[user]
        rec_u_embeddings = self.user_all_embeddings[user]

        aug_u_embeddings = F.normalize(aug_u_embeddings, p=2, dim=1)
        social_u_embeddings = F.normalize(social_u_embeddings, p=2, dim=1)
        sharing_u_embeddings = F.normalize(sharing_u_embeddings, p=2, dim=1)
        rec_u_embeddings = F.normalize(rec_u_embeddings, p=2, dim=1)

        # self-supervision prediction
        social_prediction = self.label_prediction(social_u_embeddings, aug_u_embeddings)
        sharing_prediction = self.label_prediction(sharing_u_embeddings, aug_u_embeddings)
        rec_prediction = self.label_prediction(rec_u_embeddings, aug_u_embeddings)

        # find informative positive examples for each encoder
        friend_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
        sharing_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
        rec_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)

        # neighbor-discrimination based contrastive learning
        ssl_loss = self.calculate_ssl_loss(aug_u_embeddings, friend_pos, social_u_embeddings)
        ssl_loss += self.calculate_ssl_loss(aug_u_embeddings, sharing_pos, sharing_u_embeddings)
        ssl_loss += self.calculate_ssl_loss(aug_u_embeddings, rec_pos, rec_u_embeddings)
        '''
        # L = L_r + β * L_{ssl}
        loss = rec_loss  # + self.ssl_weight * ssl_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)