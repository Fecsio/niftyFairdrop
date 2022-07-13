import torch
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F




import numpy as np
import torch
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    balanced_accuracy_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, pipeline, metrics

# Metrics
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from itertools import combinations_with_replacement

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

EPS = 1e-15


class Node2Vec(torch.nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.
    .. note::
        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.
    Args:
        edge_index (LongTensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
    """

    def __init__(
        self,
        edge_index,
        embedding_dim,
        walk_length,
        context_size,
        walks_per_node=1,
        p=1,
        q=1,
        num_negative_samples=1,
        num_nodes=None,
        sparse=False,
    ):
        super(Node2Vec, self).__init__()

        if random_walk is None:
            raise ImportError("`Node2Vec` requires `torch-cluster`.")

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to("cpu")

        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(N, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs):
        return DataLoader(
            range(self.adj.sparse_size(0)), collate_fn=self.sample, **kwargs
        )

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.adj.sparse_size(0), (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)

        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z,
        train_y,
        test_z,
        test_y,
        solver="lbfgs",
        multi_class="auto",
        *args,
        **kwargs
    ):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args, **kwargs
        ).fit(train_z, train_y)
        acc = clf.score(test_z, test_y)
        auc = metrics.roc_auc_score(test_y, clf.predict_proba(test_z)[:, 1])
        return acc, auc

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.embedding.weight.size(0),
            self.embedding.weight.size(1),
        )



def encode_classes(col):
    """
    Input:  categorical vector of any type
    Output: categorical vector of int in range 0-num_classes
    """
    classes = set(col)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, col)), dtype=np.int32)
    return labels


def onehot_classes(col):
    """
    Input:  categorical vector of int in range 0-num_classes
    Output: one-hot representation of the input vector
    """
    col2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
    col2onehot[np.arange(col.size), col] = 1
    return col2onehot


def get_edge_embeddings(z, edge_index):
    return z[edge_index[0]] * z[edge_index[1]]


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels


def train_n2v(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_rn2v(
    model, loader, optimizer, device, pos_edge_index_tr, y_aux, round1, round2, N
):

    keep = torch.where(round1, y_aux, round2)

    row, col = pos_edge_index_tr[:, keep]
    model.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to("cpu")

    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:

        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_rn2v_adaptive(
    model, loader, optimizer, device, pos_edge_index_tr, y_aux, rand, N
):

    keep = torch.where(rand, y_aux, ~y_aux)

    row, col = pos_edge_index_tr[:, keep]
    model.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to("cpu")

    model.train()

    total_loss = 0

    for pos_rw, neg_rw in loader:

        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def emb_fairness(XB, YB):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        XB, YB, test_size=0.3, stratify=YB
    )

    log = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "logi",
                    LogisticRegression(
                        multi_class="multinomial", solver="saga", max_iter=9000
                    ),
                )
            ]
        ),
        param_grid={"logi__C": [1, 10, 100]},
        cv=4,
        scoring="balanced_accuracy",
    )

    mlp = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000
                    ),
                )
            ]
        ),
        param_grid={
            "mlp__alpha": [0.001, 0.0001, 0.00001],
            "mlp__learning_rate_init": [0.01, 0.001],
        },
        cv=4,
        scoring="balanced_accuracy",
    )

    rf = model_selection.GridSearchCV(
        pipeline.Pipeline([("rf", RandomForestClassifier())]),
        param_grid={"rf__max_depth": [2, 4]},
        cv=4,
        scoring="balanced_accuracy",
    )

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    r_dict = {"RB EMB": []}
    for name, alg in c_dict.items():
        print(f"Evaluating RB with: {name}")
        alg.fit(X_train, Y_train)
        clf = alg.best_estimator_
        clf.fit(X_train, Y_train)
        score = metrics.get_scorer("balanced_accuracy")(clf, X_test, Y_test)
        r_dict["RB EMB"].append(score)

    return r_dict


def emblink_fairness(XB, YB, pos_edge_index_tr, pos_edge_index_te):

    X_train = np.hstack((XB[pos_edge_index_tr[0]], XB[pos_edge_index_tr[1]]))
    X_test = np.hstack((XB[pos_edge_index_te[0]], XB[pos_edge_index_te[1]]))
    YB = YB.reshape(-1, 1)
    Y_train = np.hstack((YB[pos_edge_index_tr[0]], YB[pos_edge_index_tr[1]]))
    Y_test = np.hstack((YB[pos_edge_index_te[0]], YB[pos_edge_index_te[1]]))

    def double_accuracy(y, y_pred, **kwargs):
        return (
            balanced_accuracy_score(y[:, 0], y_pred[:, 0])
            + balanced_accuracy_score(y[:, 1], y_pred[:, 1])
        ) / 2

    scorer = make_scorer(double_accuracy)

    log = MultiOutputClassifier(
        LogisticRegression(multi_class="multinomial", solver="saga", max_iter=1000)
    )
    mlp = MultiOutputClassifier(
        MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000)
    )
    rf = MultiOutputClassifier(RandomForestClassifier(max_depth=4))

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    r_dict = {"RB LINK": []}
    for name, alg in c_dict.items():
        print(f"Evaluating LINK RB with: {name}")
        alg.fit(X_train, Y_train)
        score = scorer(alg, X_test, Y_test)
        r_dict["RB LINK"].append(score)

    return r_dict


def fair_metrics(gt, y, group):
    metrics_dict = {
        "DPd": demographic_parity_difference(gt, y, sensitive_features=group),
        "EOd": equalized_odds_difference(gt, y, sensitive_features=group),
    }
    return metrics_dict


def prediction_fairness(test_edge_idx, test_edge_labels, te_y, group):
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))

    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)
    # MIXED DYADIC 
    
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst
    # GROUP DYADIC
    te_gd_dict = fair_metrics(
        np.concatenate([test_edge_labels, test_edge_labels], axis=0),
        np.concatenate([te_y, te_y], axis=0),
        np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
    )

    te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

    te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

    fair_list = [
        te_md_dict["DPd"],
        te_md_dict["EOd"],
        te_gd_dict["DPd"],
        te_gd_dict["EOd"],
        te_sd_dict["DPd"],
        te_sd_dict["EOd"],
    ]

    return fair_list


def link_fairness(
    Z, pos_edge_index_tr, pos_edge_index_te, neg_edge_index_tr, neg_edge_index_te, group
):

    train_edge_idx = np.concatenate([pos_edge_index_tr, neg_edge_index_tr], axis=-1)
    train_edge_embs = get_edge_embeddings(Z, train_edge_idx)
    train_edge_labels = get_link_labels(pos_edge_index_tr, neg_edge_index_tr)

    test_edge_idx = np.concatenate([pos_edge_index_te, neg_edge_index_te], axis=-1)
    test_edge_embs = get_edge_embeddings(Z, test_edge_idx)
    test_edge_labels = get_link_labels(pos_edge_index_te, neg_edge_index_te)

    log = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "logi",
                    LogisticRegression(
                        multi_class="multinomial", solver="saga", max_iter=9000
                    ),
                )
            ]
        ),
        param_grid={"logi__C": [1, 10, 100]},
        cv=4,
        scoring="balanced_accuracy",
    )

    mlp = model_selection.GridSearchCV(
        pipeline.Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000
                    ),
                )
            ]
        ),
        param_grid={
            "mlp__alpha": [0.0001, 0.00001],
            "mlp__learning_rate_init": [0.01, 0.001],
        },
        cv=4,
        scoring="balanced_accuracy",
    )

    rf = model_selection.GridSearchCV(
        pipeline.Pipeline([("rf", RandomForestClassifier())]),
        param_grid={"rf__max_depth": [2, 4]},
        cv=4,
        scoring="balanced_accuracy",
    )

    # GROUP DYADIC (one class is involved more in the generation of links)
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))
    # print(u)
    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)

    # MIXED DYADIC ( imbalanced intra-extra link creation )
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst

    c_dict = {
        "LogisticRegression": log,
        "MLPClassifier": mlp,
        "RandomForestClassifier": rf,
    }
    fair_dict = {
        "LogisticRegression": [],
        "MLPClassifier": [],
        "RandomForestClassifier": [],
    }

    for name, alg in c_dict.items():

        alg.fit(train_edge_embs, train_edge_labels)
        clf = alg.best_estimator_
        clf.fit(train_edge_embs, train_edge_labels)

        te_y = clf.predict(test_edge_embs)
        te_p = clf.predict_proba(test_edge_embs)[:, 1]

        auc = roc_auc_score(test_edge_labels, te_p)

        te_gd_dict = fair_metrics(
            np.concatenate([test_edge_labels, test_edge_labels], axis=0),
            np.concatenate([te_y, te_y], axis=0),
            np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
        )

        te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

        te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

        fair_dict[name] = [
            auc,
            # linkf,
            te_md_dict["DPd"],
            te_md_dict["EOd"],
            te_gd_dict["DPd"],
            te_gd_dict["EOd"],
            te_sd_dict["DPd"],
            te_sd_dict["EOd"],
        ]

    return fair_dict

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.lin = Linear(16, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.lin(x)
        return x

class GCN1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
        
    # def encode(self, x, pos_edge_index):
    #     x = F.relu(self.conv1(x, pos_edge_index))
    #     x = self.conv2(x, pos_edge_index)
    #     return x

    # def decode(self, z, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #     logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    #     return logits, edge_index