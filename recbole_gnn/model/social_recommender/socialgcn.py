import numpy as np
import torch
import torch.nn.functional as F

from scipy.sparse import coo_matrix, eye
from torch_geometric.utils import degree

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender
from recbole_gnn.model.layers import LightGCNConv


class SEPT(SocialRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SEPT, self).__init__(config, dataset)

        # load dataset info
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        # generate intermediate data
        self.social_edge_index, self.social_edge_weight, self.sharing_edge_index, \
        self.sharing_edge_weight = self.get_user_view_matrix(dataset)

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

        # define layer and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.user_all_embeddings = None
        self.restore_user_e = None
        self.restore_item_e = None

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
        net_keep = rand_sample(len(self._src_user), size=int(len(self._src_user) * (1 - self.drop_ratio)), replace=False)
        net_row = self._src_user[net_keep]
        net_col = self._tgt_user[net_keep]

        # concatenation and normalization
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index3 = torch.stack([net_row, net_col])
        edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        edge_weight = self.get_norm_edge_weight(edge_index, self.n_users + self.n_items)

        self.sub_graph = edge_index.to(self.device), edge_weight.to(self.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, graph=None):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]

        if graph is None:  # for the original graph
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:  # for the augmented graph
            edge_index, edge_weight = graph

        for _ in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
            ##norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return user_all_embeddings, item_all_embeddings

    def user_view_forward(self):
        all_social_embeddings = self.user_embedding.weight
        all_sharing_embeddings = self.user_embedding.weight
        social_embeddings_list = [all_social_embeddings]
        sharing_embeddings_list = [all_sharing_embeddings]

        for _ in range(self.n_layers):
            # friend view
            all_social_embeddings = self.gcn_conv(all_social_embeddings, self.social_edge_index, self.social_edge_weight)
            #norm_social_embeddings = F.normalize(all_social_embeddings, p=2, dim=1)
            social_embeddings_list.append(all_social_embeddings)
            # sharing view
            all_sharing_embeddings = self.gcn_conv(all_sharing_embeddings, self.sharing_edge_index, self.sharing_edge_weight)
            #norm_sharing_embeddings = F.normalize(all_sharing_embeddings, p=2, dim=1)
            sharing_embeddings_list.append(all_sharing_embeddings)

        social_all_embeddings = torch.stack(social_embeddings_list, dim=1)
        social_all_embeddings = torch.sum(social_all_embeddings, dim=1)

        sharing_all_embeddings = torch.stack(sharing_embeddings_list, dim=1)
        sharing_all_embeddings = torch.sum(sharing_all_embeddings, dim=1)

        return social_all_embeddings, sharing_all_embeddings

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

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings,require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_loss(self, interaction):
        # preference view
        rec_loss = self.calculate_rec_loss(interaction)

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

        # L = L_r + β * L_{ssl}
        loss = rec_loss + self.ssl_weight * ssl_loss

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