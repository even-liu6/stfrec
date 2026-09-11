import torch
from torch import nn
import numpy as np
import scipy.sparse as sp

import utility.trainer as trainer
import utility.tools as tools
import utility.losses as losses


class STFRec(nn.Module):
    def __init__(self, args, dataset, device):
        super().__init__()

        self.model_name = "stfrec"
        self.dataset = dataset
        self.args = args
        self.device = device

        self.reg_lambda = float(self.args.reg_lambda)
        self.ssl_lambda = float(self.args.ssl_lambda)
        self.tau = float(self.args.tau)
        self.encoder = self.args.encoder

        self.rw_alpha = float(self.args.rw_alpha)
        self.rw_topk = int(self.args.rw_topk)
        self.rw_beta = float(self.args.rw_beta)
        self.rw_candidate_k = int(getattr(self.args, "rw_candidate_k", 300))
        self.rw_user_norm = int(getattr(self.args, "rw_user_norm", 1))
        self.rw_pop_penalty = int(getattr(self.args, "rw_pop_penalty", 1))
        self.rw_score_norm = str(getattr(self.args, "rw_score_norm", "minmax"))

        self.current_epoch = 0
        self.rw_decay_start = int(getattr(self.args, "rw_decay_start", 1000000))
        self.rw_decay_rate = float(getattr(self.args, "rw_decay_rate", 1.0))
        self.rw_min_ratio = float(getattr(self.args, "rw_min_ratio", 1.0))

        self.activation = nn.Sigmoid()

        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.dataset.num_users,
            embedding_dim=int(self.args.embedding_size)
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.dataset.num_items,
            embedding_dim=int(self.args.embedding_size)
        )

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.adj_mat = self.dataset.sparse_adjacency_matrix()
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

        self.train_mat_csr = self.dataset.train_mat.tocsr().astype(np.float32)

        self.item_transition = None

        if self.rw_alpha > 0:
            print("\t Building item-item structural diffusion graph...")
            self.item_transition = self._build_item_transition().to(self.device)
            print("\t Item-item structural diffusion graph ready.")

    def _topk_sparse_rows(self, mat, k):
        mat = mat.tocsr()

        rows = []
        cols = []
        vals = []

        for r in range(mat.shape[0]):
            start = mat.indptr[r]
            end = mat.indptr[r + 1]

            row_cols = mat.indices[start:end]
            row_vals = mat.data[start:end]

            if len(row_vals) == 0:
                continue

            if len(row_vals) > k:
                idx = np.argpartition(row_vals, -k)[-k:]
                row_cols = row_cols[idx]
                row_vals = row_vals[idx]

            rows.extend([r] * len(row_vals))
            cols.extend(row_cols.tolist())
            vals.extend(row_vals.tolist())

        return sp.csr_matrix(
            (
                np.asarray(vals, dtype=np.float32),
                (
                    np.asarray(rows, dtype=np.int64),
                    np.asarray(cols, dtype=np.int64)
                )
            ),
            shape=mat.shape,
            dtype=np.float32
        )

    def _build_item_transition(self):
        """
        Structural diffusion graph with ablation switches.

        Full:
            C = R^T D_u^{-1} R

        w/o user-degree norm:
            C = R^T R

        Then optionally:
            target-item popularity penalty
            top-k sparse pruning
            row normalization
        """
        R = self.train_mat_csr

        if self.rw_user_norm == 1:
            user_deg = np.asarray(R.sum(axis=1)).flatten().astype(np.float32)

            with np.errstate(divide="ignore"):
                user_inv = np.power(user_deg, -1.0)

            user_inv[np.isinf(user_inv)] = 0.0
            user_inv[np.isnan(user_inv)] = 0.0

            Pui = sp.diags(user_inv).dot(R).tocsr()
        else:
            # Ablation: remove user-degree normalization.
            # C becomes ordinary item-item co-occurrence R^T R.
            Pui = R

        C = R.T.dot(Pui).tocsr().astype(np.float32)

        C.setdiag(0.0)
        C.eliminate_zeros()

        if self.rw_pop_penalty == 1 and self.rw_beta > 0:
            item_deg = np.asarray(R.sum(axis=0)).flatten().astype(np.float32)
            target_penalty = np.power(item_deg + 1.0, -self.rw_beta).astype(np.float32)
            C = C.multiply(target_penalty.reshape(1, -1)).tocsr()

        C = self._topk_sparse_rows(C, self.rw_topk)

        row_sum = np.asarray(C.sum(axis=1)).flatten().astype(np.float32)

        with np.errstate(divide="ignore"):
            row_inv = np.power(row_sum, -1.0)

        row_inv[np.isinf(row_inv)] = 0.0
        row_inv[np.isnan(row_inv)] = 0.0

        S = sp.diags(row_inv).dot(C).tocsr()

        return tools.convert_sp_mat_to_sp_tensor(S)

    def _batch_history_dense(self, user):
        users_np = user.detach().cpu().numpy().astype(np.int64)

        H = self.train_mat_csr[users_np].astype(np.float32)

        row_sum = np.asarray(H.sum(axis=1)).flatten().astype(np.float32)

        with np.errstate(divide="ignore"):
            row_inv = np.power(row_sum, -1.0)

        row_inv[np.isinf(row_inv)] = 0.0
        row_inv[np.isnan(row_inv)] = 0.0

        H = sp.diags(row_inv).dot(H).tocsr()

        return torch.from_numpy(H.toarray()).float().to(self.device)

    def get_effective_rw_alpha(self):
        if self.rw_alpha <= 0:
            return 0.0

        epoch = int(getattr(self, "current_epoch", 0))

        if epoch <= 0:
            return self.rw_alpha

        if epoch <= self.rw_decay_start:
            return self.rw_alpha

        decay_steps = epoch - self.rw_decay_start
        ratio = self.rw_decay_rate ** decay_steps
        ratio = max(self.rw_min_ratio, ratio)

        return self.rw_alpha * ratio

    def aggregate(self):
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0
        )

        all_embeddings = [embeddings]

        for layer in range(int(self.args.gcn_layer)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(
            final_embeddings,
            [self.dataset.num_users, self.dataset.num_items]
        )

        return user_emb, item_emb

    def forward(self, user, positive, negative):
        if self.encoder == "MF":
            all_user_gcn_embed = self.user_embedding.weight
            all_item_gcn_embed = self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]
        positive_gcn_embed = all_item_gcn_embed[positive.long()]
        negative_gcn_embed = all_item_gcn_embed[negative.long()]

        user_embed = self.user_embedding(user)
        positive_embed = self.item_embedding(positive)
        negative_embed = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(
            user_gcn_embed,
            positive_gcn_embed,
            negative_gcn_embed
        )

        reg_loss = losses.get_reg_loss(
            user_embed,
            positive_embed,
            negative_embed
        ) * self.reg_lambda

        na_loss = losses.get_neighbor_aggregate_loss(
            user_gcn_embed,
            positive_gcn_embed,
            self.tau
        ) * self.ssl_lambda

        loss_list = [bpr_loss, reg_loss, na_loss]

        return loss_list

    def get_rating_for_test(self, user):
        """
        Structural diffusion fusion with ablation switch.

        Full:
            diffusion_score = user-wise min-max(H_u S)
            score = base_score + alpha * diffusion_score

        w/o min-max:
            score = base_score + alpha * raw_diffusion_score
        """
        if self.encoder == "MF":
            all_user_gcn_embed = self.user_embedding.weight
            all_item_gcn_embed = self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]

        base_score = torch.matmul(
            user_gcn_embed,
            all_item_gcn_embed.t()
        )

        effective_alpha = self.get_effective_rw_alpha()

        if effective_alpha > 0 and self.item_transition is not None:
            H = self._batch_history_dense(user)

            diffusion_score = torch.sparse.mm(
                self.item_transition.transpose(0, 1),
                H.t()
            ).t()

            if self.rw_score_norm == "minmax":
                diff_min = diffusion_score.min(dim=1, keepdim=True).values
                diff_max = diffusion_score.max(dim=1, keepdim=True).values

                diffusion_score = (
                    diffusion_score - diff_min
                ) / (
                    diff_max - diff_min + 1e-8
                )
            elif self.rw_score_norm == "none":
                diffusion_score = diffusion_score
            else:
                raise ValueError("Unknown rw_score_norm: {}".format(self.rw_score_norm))

            score = base_score + effective_alpha * diffusion_score
        else:
            score = base_score

        rating = self.activation(score)

        return rating

class Trainer():
    def __init__(self, args, dataset, device, logger):
        self.model = STFRec(args, dataset, device)
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger

    def train(self):
        trainer.training(
            self.model,
            self.args,
            self.dataset,
            self.device,
            self.logger
        )
