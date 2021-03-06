import os
from algorithms.EMGCN.embedding_model import EM_GCN, GAT
from algorithms.EMGCN.stable_factor import StableFactor
from algorithms.network_alignment_model import NetworkAlignmentModel

from utils.graph_utils import load_gt
import torch.nn.functional as F

from algorithms.EMGCN.utils import cv_coo_sparse, get_acc, normalize_numpy, get_similarity_matrices, get_numpy_simi_matrix, get_cosine, get_dict_from_S, linkpred_loss_multiple_layer, supervised_loss_multiple_layer
from scipy import sparse as sp
import torch
import numpy as np

import time
from tqdm import tqdm
from numpy import *
import torch
import torch.nn as nn


class EMGCN(NetworkAlignmentModel):
    """
    EMGCN model for Knowledge Graph Alignment task
    """

    def __init__(self, source_dataset, target_dataset, args):
        """
        :params source_dataset: source graph
        :params target_dataset: target graph
        :params args: more config params
        """
        super(EMGCN, self).__init__(source_dataset, target_dataset)
        self.args = args
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alpha_att_val = [args.rel, args.att, args.attval]
        self.n_node_s = len(self.source_dataset.G.nodes())
        self.n_node_t = len(self.target_dataset.G.nodes())
        self.full_dict = load_gt(
            args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.alphas = [1, 1, 1, 1, 1, 1]
        self.att_dict1, self.att_dict2 = self.source_dataset.get_raw_att_dicts()
        self.source_att_set = set(self.att_dict1.keys())
        self.target_att_set = set(self.att_dict2.keys())
        self.kept_att = self.source_att_set.intersection(self.target_att_set)
        self.att_dict_inverse1 = {v: k for k, v in self.att_dict1.items()}
        self.att_dict_inverse2 = {v: k for k, v in self.att_dict2.items()}
        self.source_att_value = self.source_dataset.get_the_raw_datastructure(
            self.source_dataset.ent_att_val1, self.att_dict_inverse1, self.kept_att)
        self.target_att_value = self.source_dataset.get_the_raw_datastructure(
            self.source_dataset.ent_att_val2, self.att_dict_inverse2, self.kept_att)
        self.statistic()

    def count_att_val(self, data):
        count = 0
        for key, value in data.items():
            count += len(value['att'])
        return count

    def statistic(self):
        print("Number of filted att source: {}".format(len(self.source_att_set)))
        print("Number of filted att target: {}".format(len(self.target_att_set)))
        print("Number of att dict: {}".format(len(self.kept_att)))
        print("Number of att triple source: {}".format(
            self.count_att_val(self.source_att_value)))
        print("Number of att triple target: {}".format(
            self.count_att_val(self.target_att_value)))

    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        # sparse
        start_A_hat = time.time()
        source_A_hat = self.source_dataset.construct_laplacian(
            sparse=self.args.sparse, direct=self.args.direct_adj, typee="new")
        target_A_hat = self.target_dataset.construct_laplacian(
            sparse=self.args.sparse, direct=self.args.direct_adj, typee="new")
        source_A_hat_sym = self.source_dataset.construct_laplacian(
            sparse=self.args.sparse, direct=False, typee="new")
        target_A_hat_sym = self.target_dataset.construct_laplacian(
            sparse=self.args.sparse, direct=False, typee="new")

        if sp.issparse(source_A_hat):
            source_A_hat = cv_coo_sparse(source_A_hat)
            target_A_hat = cv_coo_sparse(target_A_hat)
            source_A_hat_sym = cv_coo_sparse(source_A_hat_sym)
            target_A_hat_sym = cv_coo_sparse(target_A_hat_sym)
        else:
            source_A_hat = torch.FloatTensor(source_A_hat)
            target_A_hat = torch.FloatTensor(target_A_hat)
            source_A_hat_sym = torch.FloatTensor(source_A_hat_sym)
            target_A_hat_sym = torch.FloatTensor(target_A_hat_sym)

        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()
            source_A_hat_sym = source_A_hat_sym.cuda()
            target_A_hat_sym = target_A_hat_sym.cuda()

        print("Create A hat time: {:.4f}".format(time.time() - start_A_hat))

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        # Norm2 normalization
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats, source_A_hat_sym, target_A_hat_sym

    def get_simi_att(self):
        simi = np.zeros((self.n_node_s, self.n_node_t))
        value_simi = np.zeros((self.n_node_s, self.n_node_t))
        i = 0
        for snode in tqdm(self.source_att_value):
            i += 1
            # if we're testing
            if i > 100 and self.args.emb_epochs <= 1 and self.args.refinement_epochs <= 1:
                break
            # print('snode', snode)
            if len(self.source_att_value[snode]['att']) == 0:
                continue
            snode_index = self.source_dataset.id2idx[snode]
            snode_att = self.source_att_value[snode]['att']
            # breakpoint()
            for tnode in self.target_att_value:
                if len(self.target_att_value[tnode]['att']) == 0:
                    continue
                tnode_index = self.target_dataset.id2idx[tnode]
                tnode_att = self.target_att_value[tnode]['att']
                common_att = snode_att.intersection(tnode_att)
                # TODO: Modify the tradeoff from 0.5 to 0.2 and 0.8 ...
                tradeoff = 0.5
                common_att = self.source_dataset.cluster_common_attributes(
                    snode_att, tnode_att, common_att, tradeoff)
                value_simi_this = 0
                for ele in common_att:
                    source_values = self.source_att_value[snode]['att_value'][ele]
                    target_values = self.target_att_value[tnode]['att_value'][ele]
                    value_simi_this += self.source_dataset.embedded_word_simi(
                        source_values, target_values)
                if value_simi_this > 0:
                    value_simi_this /= len(common_att)
                value_simi[snode_index, tnode_index] = value_simi_this
                # attribute similarity matrix is simi
                simi[snode_index, tnode_index] = self.source_dataset.embedded_word_simi(
                    snode_att, tnode_att)
        return simi, value_simi

    def score(self, dictt):
        source_edges = self.source_dataset.G.edges()
        # target_edges = self.target_dataset.G.edges()
        count_true = 0
        for edge in source_edges:
            target_edge = (dictt[edge[0]], dictt[edge[1]])
            if self.target_dataset.G.has_edge(*target_edge):
                count_true += 1
        return count_true / len(source_edges)

    def get_candidate(self, source_outputs, target_outputs, num_stable):
        List_S = get_similarity_matrices(source_outputs, target_outputs)
        #List_S_2 = [self.att_simi_matrix, self.value_simi_matrix]

        source_candidates = []
        target_candidates = []
        count_true_candidates = 0
        if len(List_S) < 2:
            print(
                "The current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

        num_source_nodes = self.n_node_s
        num_target_nodes = self.n_node_t

        def k_largest_index_argsort(a, k):
            idx = np.argsort(a.ravel())[:-k-1:-1]
            return np.column_stack(np.unravel_index(idx, a.shape))

        stable_candidates_layer0 = k_largest_index_argsort(
            List_S[0], num_stable)
        source_candidates = stable_candidates_layer0[:, 0]
        target_candidates = stable_candidates_layer0[:, 1]
        for i in range(1, len(List_S)):
            source_candidate_layer_i = []
            target_candidate_layer_i = []
            S_i = List_S[i]
            for k in range(len(source_candidates)):
                source_k = source_candidates[k]
                target_k = target_candidates[k]
                if S_i[source_k].argmax() == target_k:
                    source_candidate_layer_i.append(source_k)
                    target_candidate_layer_i.append(target_k)
            source_candidates = source_candidate_layer_i
            target_candidates = target_candidate_layer_i

        count_true_candidates = 0
        for i in range(i, len(source_candidates)):
            try:
                if self.full_dict[source_candidates[i]] == target_candidates[i]:
                    count_true_candidates += 1
            except:
                continue
        print("Num candidates: {}, num true candidates: {}".format(
            len(source_candidates), count_true_candidates))
        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

    def refine(self, embedding_model, refinement_model, source_A_hat, target_A_hat, att_value_simi_matrix):
        # INIT BEFORE LOOP

        embedding_model.eval()  # stops runtimeerror ... cpu backend

        source_outputs = embedding_model(source_A_hat, 's')
        target_outputs = embedding_model(target_A_hat, 't')
        acc, self.S, _ = get_acc(source_outputs, target_outputs, self.alphas)
        print("ACC: {}".format(acc))

        dictt = get_dict_from_S(
            self.S, self.source_dataset.id2idx, self.target_dataset.id2idx)
        score = self.score(dictt)
        score_max = score

        for epoch in tqdm(range(self.args.refinement_epochs)):
            print("Refinement epoch: {}".format(epoch))
            source_candidates, target_candidates = self.get_candidate(
                source_outputs, target_outputs, (epoch+1) * self.args.num_each_refine)
            refinement_model.alpha_source[source_candidates] *= self.args.point
            refinement_model.alpha_target[target_candidates] *= self.args.point

            if not self.args.attention:
                X = refinement_model(source_A_hat, 's')
                source_outputs = embedding_model(X, 's')
                X = refinement_model(target_A_hat, 't')
                target_outputs = embedding_model(X, 't')
            else:
                # problem is there is a bug somewhere so that dense matrices do not process in emb_model
                # but sparse do not work in refinement model...
                X = refinement_model(source_A_hat.to_dense(), 's')
                source_outputs = embedding_model(X.to_sparse(), 's')
                X = refinement_model(target_A_hat.to_dense(), 't')
                target_outputs = embedding_model(X.to_sparse(), 't')

            acc, S, _ = get_acc(source_outputs, target_outputs, self.alphas)
            print(acc)
            score = np.max(S, axis=1).mean()
            dictt = get_dict_from_S(
                S, self.source_dataset.id2idx, self.target_dataset.id2idx)
            score = self.score(dictt)
            if score > score_max:
                score_max = score
                self.S = S

        self.S = self.alpha_att_val[0] * self.S + \
            self.alpha_att_val[1] * att_value_simi_matrix

        return self.S

    def train_embedding(self, embedding_model, refinement_model, source_A_hat, target_A_hat, structural_optimizer, source_A_hat_sym, target_A_hat_sym):
        for epoch in tqdm(range(self.args.emb_epochs)):
            start_time = time.time()
            print("Structure learning epoch: {}".format(epoch))
            for i in range(2):
                structural_optimizer.zero_grad()
                if i == 0:
                    A_hat_sym = source_A_hat_sym
                    outputs = embedding_model(source_A_hat, 's')[1:]
                    # print(len(outputs))
                else:
                    A_hat_sym = target_A_hat_sym
                    outputs = embedding_model(target_A_hat, 't')[1:]
                    # print(outputs)
                loss = linkpred_loss_multiple_layer(
                    outputs, A_hat_sym, self.args.cuda)
                if self.args.log:
                    print("Loss: {:.4f}".format(loss.data))
                loss.backward()
                structural_optimizer.step()
        print("Epoch time: {:.4f}".format(time.time() - start_time))

        print("Done structural training")

        embedding_model.eval()
        self.att_simi_matrix, self.value_simi_matrix = self.get_simi_att()
        att_value_simi_matrix = self.att_simi_matrix + self.value_simi_matrix
        if not self.args.attention:
            source_A_hat = source_A_hat.to_dense()
            target_A_hat = target_A_hat.to_dense()
        # refinement

        self.refine(embedding_model, refinement_model,
                    source_A_hat, target_A_hat, att_value_simi_matrix)

    """
    incorporate supervised and unsupervised (train_data ground truth + EMGCN method)
    """

    def train_embedding_hybrid(self, embedding_model, refinement_model, source_A_hat, target_A_hat,
                               structural_optimizer, source_A_hat_sym,
                               target_A_hat_sym, train_data, loss_split):

        t = len(train_data)
        # k is the number of negative samples to generate per gt sample
        k = 3

        L = np.ones((t, k)) * train_data[:, 0].reshape((t, 1))
        neg_left = L.reshape((t*k,))

        for epoch in tqdm(range(self.args.emb_epochs)):
            start_time = time.time()
            print("Structure learning epoch: {}".format(epoch))

            # from gcn align paper: generate new negative samples every 10 epochs
            if epoch % 10 == 0:
                neg_right = np.random.choice(train_data[:, 1], t * k)

            structural_optimizer.zero_grad()
            source_outputs = embedding_model(source_A_hat, 's')[1:]
            target_outputs = embedding_model(target_A_hat, 't')[1:]

            ''' Uncomment these  lines to use just last GCN layer for memory debugging'''
            # source_outputs = embedding_model(source_A_hat, 's')[-1]
            # target_outputs = embedding_model(target_A_hat, 't')[-1]

            supervised_loss = supervised_loss_multiple_layer(
                source_outputs, target_outputs, train_data, neg_left, neg_right, k, self.args.cuda)

            # Only difference is we compute the loss + update gradient at the same time
            # and not first doing source then target
            source_unsupervised_loss = linkpred_loss_multiple_layer(
                source_outputs, source_A_hat_sym, self.args.cuda)
            target_unsupervised_loss = linkpred_loss_multiple_layer(
                target_outputs, target_A_hat_sym, self.args.cuda)

            loss = supervised_loss/2 + \
                (source_unsupervised_loss + target_unsupervised_loss)/2
            # TODO: Try new loss split
            if False:
                loss = loss_split * (supervised_loss/2) + (1 - loss_split) * \
                    (source_unsupervised_loss + target_unsupervised_loss) / 2

            print("Loss: {:.4f}".format(loss.data))
            loss.backward()
            structural_optimizer.step()
            print("Epoch time: {:.4f}".format(time.time() - start_time))

        print("Done structural training")

        embedding_model.eval()
        self.att_simi_matrix, self.value_simi_matrix = self.get_simi_att()
        att_value_simi_matrix = self.att_simi_matrix + self.value_simi_matrix
        if not self.args.attention:
            source_A_hat = source_A_hat.to_dense()
            target_A_hat = target_A_hat.to_dense()
        # refinement

        self.refine(embedding_model, refinement_model,
                    source_A_hat, target_A_hat, att_value_simi_matrix)

    """
    added parameter for train_data ground truth
    """

    def train_embedding_supervised(self, embedding_model, refinement_model, source_A_hat, target_A_hat, structural_optimizer, train_data):

        t = len(train_data)
        # k is the number of negative samples to generate per gt sample
        k = 3

        L = np.ones((t, k)) * train_data[:, 0].reshape((t, 1))
        neg_left = L.reshape((t*k,))

        for epoch in tqdm(range(self.args.emb_epochs)):
            start_time = time.time()
            print("Structure learning epoch: {}".format(epoch))

            # from gcn align paper: generate new negative samples every 10 epochs
            if epoch % 10 == 0:
                neg_right = np.random.choice(train_data[:, 1], t * k)

            structural_optimizer.zero_grad()
            source_outputs = embedding_model(source_A_hat, 's')[1:]
            target_outputs = embedding_model(target_A_hat, 't')[1:]

            ''' Uncomment these  lines to use just last GCN layer for memory debugging'''
            # source_outputs = embedding_model(source_A_hat, 's')[-1]
            # target_outputs = embedding_model(target_A_hat, 't')[-1]

            loss = supervised_loss_multiple_layer(
                source_outputs, target_outputs, train_data, neg_left, neg_right, k, self.args.cuda)
            # if self.args.log:
            #
            print("Loss: {:.4f}".format(loss.data))
            loss.backward()
            structural_optimizer.step()
        print("Epoch time: {:.4f}".format(time.time() - start_time))

        print("Done structural training")

        embedding_model.eval()
        self.att_simi_matrix, self.value_simi_matrix = self.get_simi_att()
        att_value_simi_matrix = self.att_simi_matrix + self.value_simi_matrix
        if not self.args.attention:
            source_A_hat = source_A_hat.to_dense()
            target_A_hat = target_A_hat.to_dense()
        # refinement

        self.refine(embedding_model, refinement_model,
                    source_A_hat, target_A_hat, att_value_simi_matrix)

    """
    Added train_data parameter to allow for supervised 
    """

    def align(self, train_data=None):
        source_A_hat, target_A_hat, source_feats, target_feats, source_A_hat_sym, target_A_hat_sym = self.get_elements()

        # print(source_feats.size())
        attention = self.args.attention

        if attention:
            # , setting refine epochs to 0")
            print("using new attention network")
            #self.args.refinement_epochs = 0
            embedding_model = GAT(
                activate_function=self.args.act,
                num_GCN_blocks=self.args.num_GCN_blocks,
                output_dim=self.args.embedding_dim,
                num_source_nodes=self.n_node_s,
                num_target_nodes=self.n_node_t,
                source_feats=source_feats,
                target_feats=target_feats,
                direct=self.args.direct_adj,
            )
        else:
            embedding_model = EM_GCN(
                activate_function=self.args.act,
                num_GCN_blocks=self.args.num_GCN_blocks,
                output_dim=self.args.embedding_dim,
                num_source_nodes=self.n_node_s,
                num_target_nodes=self.n_node_t,
                source_feats=source_feats,
                target_feats=target_feats,
                direct=self.args.direct_adj,
            )

        refinement_model = StableFactor(
            self.n_node_s, self.n_node_t, self.args.cuda)

        if self.args.cuda:
            embedding_model = embedding_model.cuda()
        structural_optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, embedding_model.parameters()), lr=self.args.lr)
        embedding_model.train()

        """
        train_data is ground truth samples
        GCN-align paper uses SGD instead of Adam
        """
        if train_data is not None:
            print("Supervised")
            hybrid = True
            loss_split = 0.2

            if hybrid:
                self.train_embedding_hybrid(embedding_model, refinement_model, source_A_hat,
                                            target_A_hat, structural_optimizer, source_A_hat_sym, target_A_hat_sym,
                                            train_data, loss_split)
            else:
                self.train_embedding_supervised(embedding_model, refinement_model, source_A_hat,
                                                target_A_hat, structural_optimizer, train_data)
        else:
            self.train_embedding(embedding_model, refinement_model, source_A_hat,
                                 target_A_hat, structural_optimizer, source_A_hat_sym, target_A_hat_sym)
        """"""

        return self.S
