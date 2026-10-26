"""
Implementation of functions to compute our joint DR/clustering score.
"""

import torch
from src.utils import plan_color
from torchmetrics.clustering import (
    AdjustedRandScore,
    NormalizedMutualInfoScore,
    AdjustedMutualInfoScore,
    HomogeneityScore,
)
from src.utils_hyperbolic import minkowski_ip2, lorentz_to_poincare, log_poincare
import random
from tqdm import tqdm

from geoopt.optim import RiemannianAdam
from src.affinities import NanError


# %%
class silhouette_coefficients:
    @staticmethod
    def score(X, labels, loss=False, hyperbolic=False):
        """Compute Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.
        Code developed in NumPy by Alexandre Abraham:
        https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as
                torch tensor without moving it to the CPU. Can therefore
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float
                on CPU will be returned.
        Returns
        -------
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        References
        ----------
        Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis". Computational
            and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
        http://en.wikipedia.org/wiki/Silhouette_(clustering)
        """
        # if type(labels) != type(torch.HalfTensor()):
        #     labels = torch.HalfTensor(labels)
        # if not labels.is_cuda:
        #     labels = labels.cuda()
        # if type(X) != type(torch.HalfTensor()):
        #     X = torch.HalfTensor(X)
        # if not X.is_cuda:
        #     X = X.cuda()
        unique_labels = torch.unique(labels)
        A = silhouette_coefficients._intra_cluster_distances_block(
            X, labels, unique_labels, hyperbolic
        )
        B = silhouette_coefficients._nearest_cluster_distance_block(
            X, labels, unique_labels, hyperbolic
        )
        sil_samples = (B - A) / torch.maximum(A, B)
        return torch.nan_to_num(sil_samples)

    @staticmethod
    def _intra_cluster_distances_block(X, labels, unique_labels, hyperbolic):
        """Calculate the mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = torch.zeros(labels.size(), dtype=X.dtype, device=X.device)
        values = [
            silhouette_coefficients._intra_cluster_distances_block_(
                X[torch.where(labels == label)[0]], hyperbolic
            )
            for label in unique_labels
        ]
        for label, values_ in zip(unique_labels, values):
            intra_dist[torch.where(labels == label)[0]] = values_
        return intra_dist

    @staticmethod
    def _intra_cluster_distances_block_(subX, hyperbolic):
        if hyperbolic:
            distances = torch.arccosh(torch.clamp(-minkowski_ip2(subX, subX), min=1 + 1e-15))
        else:
            distances = torch.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)

    @staticmethod
    def _nearest_cluster_distance_block(X, labels, unique_labels, hyperbolic):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = torch.full(labels.size(), torch.inf, dtype=X.dtype, device=X.device)
        # Compute cluster distance between pairs of clusters
        label_combinations = torch.combinations(unique_labels, 2)
        values = [
            silhouette_coefficients._nearest_cluster_distance_block_(
                X[torch.where(labels == label_a)[0]],
                X[torch.where(labels == label_b)[0]],
                hyperbolic,
            )
            for label_a, label_b in label_combinations
        ]
        for (label_a, label_b), (values_a, values_b) in zip(label_combinations, values):
            indices_a = torch.where(labels == label_a)[0]
            inter_dist[indices_a] = torch.minimum(values_a, inter_dist[indices_a])
            del indices_a
            indices_b = torch.where(labels == label_b)[0]
            inter_dist[indices_b] = torch.minimum(values_b, inter_dist[indices_b])
            del indices_b
        return inter_dist

    @staticmethod
    def _nearest_cluster_distance_block_(subX_a, subX_b, hyperbolic):
        if hyperbolic:
            dist = torch.arccosh(torch.clamp(-minkowski_ip2(subX_a, subX_b), min=1 + 1e-15))
        else:
            dist = torch.cdist(subX_a, subX_b)
        dist_a = dist.mean(axis=1)
        dist_b = dist.mean(axis=0)
        return dist_a, dist_b


class weighted_silhouette_coefficients:
    @staticmethod
    def score(X, p, labels, loss=False, hyperbolic=False):
        """Compute Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.
        Code developed in NumPy by Alexandre Abraham:
        https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        p: array [n_samples_a]
            propability vector taking into account the relative importance of samples
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as
                torch tensor without moving it to the CPU. Can therefore
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float
                on CPU will be returned.
        Returns
        -------
        silhouette_coefficients : float
            Weighted Silhouette Coefficient for all samples.
        References
        ----------
        """
        # if type(labels) != type(torch.HalfTensor()):
        #     labels = torch.HalfTensor(labels)
        # if not labels.is_cuda:
        #     labels = labels.cuda()
        # if type(X) != type(torch.HalfTensor()):
        #     X = torch.HalfTensor(X)
        # if not X.is_cuda:
        #     X = X.cuda()
        assert (
            torch.sum(p <= 0.0) == 0.0
        )  # support only strictly positive probability weights for simplicity
        unique_labels = torch.unique(labels)
        A = weighted_silhouette_coefficients._intra_cluster_distances_block(
            X, p, labels, unique_labels, hyperbolic
        )
        B = weighted_silhouette_coefficients._nearest_cluster_distance_block(
            X, p, labels, unique_labels, hyperbolic
        )
        sil_samples = (B - A) / torch.maximum(A, B)
        return torch.nan_to_num(sil_samples)

    @staticmethod
    def _intra_cluster_distances_block(X, p, labels, unique_labels, hyperbolic):
        """Calculate the weighted mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        p: array [n_samples_a]
            propability vector taking into account the relative importance of samples
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = torch.zeros(labels.size(), dtype=X.dtype, device=X.device)
        for label in unique_labels:
            idx_labels = torch.argwhere(labels == label)[:, 0]
            values = weighted_silhouette_coefficients._intra_cluster_distances_block_(
                X[idx_labels], p[idx_labels], hyperbolic
            )
            intra_dist[idx_labels] = values

        return intra_dist

    @staticmethod
    def _intra_cluster_distances_block_(subX, subp, hyperbolic=False):
        subn = subp.shape[0]
        subp_mat = subp.view(1, -1).repeat((subn, 1))
        if hyperbolic:
            distances = torch.arccosh(torch.clamp(-minkowski_ip2(subX, subX), min=1 + 1e-15))
        else:
            distances = torch.cdist(subX, subX)
        weighted_distances = distances * subp_mat
        subp_math = subp_mat.fill_diagonal_(0.0)
        return weighted_distances.sum(axis=1) / subp_math.sum(
            axis=1
        )  # as for the silhouette score we do not take into account p[i]d(xi, xi)

    @staticmethod
    def _nearest_cluster_distance_block(X, p, labels, unique_labels, hyperbolic):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        p: array [n_samples_a]
            propability vector taking into account the relative importance of samples
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = torch.full(labels.size(), torch.inf, dtype=X.dtype, device=X.device)
        # Compute cluster distance between pairs of clusters
        label_combinations = torch.combinations(unique_labels, 2)
        values = []
        idx_labels = []
        for label_a, label_b in label_combinations:
            idx_labels_a = torch.argwhere(labels == label_a)[:, 0]
            idx_labels_b = torch.argwhere(labels == label_b)[:, 0]
            dist_a, dist_b = weighted_silhouette_coefficients._nearest_cluster_distance_block_(
                X[idx_labels_a],
                p[idx_labels_a],
                X[idx_labels_b],
                p[idx_labels_b],
                hyperbolic,
            )
            values.append((dist_a, dist_b))
            idx_labels.append((idx_labels_a, idx_labels_b))

        for (
            (label_a, label_b),
            (values_a, values_b),
            (idx_labels_a, idx_labels_b),
        ) in zip(label_combinations, values, idx_labels):
            # indices_a = torch.where(labels == label_a)[0]
            inter_dist[idx_labels_a] = torch.minimum(values_a, inter_dist[idx_labels_a])
            inter_dist[idx_labels_b] = torch.minimum(values_b, inter_dist[idx_labels_b])

        return inter_dist

    @staticmethod
    def _nearest_cluster_distance_block_(subX_a, subp_a, subX_b, subp_b, hyperbolic):
        if hyperbolic:
            dist = torch.arccosh(torch.clamp(-minkowski_ip2(subX_a, subX_b), min=1 + 1e-15))
        else:
            dist = torch.cdist(subX_a, subX_b)
        weighted_dist_a = dist * subp_b[None, :]
        weighted_dist_b = dist * subp_a[:, None]
        norm_weighted_dist_a = weighted_dist_a.sum(dim=1) / subp_b.sum()
        norm_weighted_dist_b = weighted_dist_b.sum(dim=0) / subp_a.sum()

        return norm_weighted_dist_a, norm_weighted_dist_b


# %%


def knn_recall(X, Z, T, k=10):
    N = X.shape[0]
    also_N = T.shape[0]
    assert N == also_N, "X and T should have the same number of elements"

    # -- Kary neighborhood for X --
    Cx = torch.cdist(X, X)
    indices_NN_X = torch.topk(Cx, k=k, dim=1, largest=False).indices

    row_indices = (torch.arange(indices_NN_X.size(0)).unsqueeze(1).expand_as(indices_NN_X)).to(
        X.device
    )
    row_col_pairs_input = torch.stack((row_indices.flatten(), indices_NN_X.flatten()), dim=1)

    # -- optimistic Kary neighborhood for Z --
    Z_ = T @ Z

    Cz = torch.cdist(Z_, Z_)
    # we retrieve the neighbors whose distance is inferior to the limit
    # to be optimistic
    limit_value = torch.topk(Cz, k=k, dim=1, largest=False, sorted=True).values[:, -1]
    row, col = torch.where(Cz <= limit_value[:, None])
    row_col_pairs_output = torch.stack((row, col), dim=1)

    # Find common pairs in input and output
    common_pairs = common_elements(row_col_pairs_input, row_col_pairs_output)
    N_common = len(common_pairs)

    score = N_common / (N * k)
    return score


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]


def kmeans_score(Z, T, Y, weighted=False, hyperbolic=False):
    print(f"--- Z: {Z.shape} ---")
    assert (T.sum(-1) - 1).sum() < 1e-6, "T should be a membership matrix."

    n_labels = min(len(torch.unique(Y)), Z.shape[0])
    from sklearn.cluster import KMeans

    sample_weight = None
    if hyperbolic:
        # project from lorentz to poincare and compute log map
        # centers = torch.zeros(Z.shape, device=Z.device, dtype=Z.dtype)
        # local_Z = lorentz_to_poincare(Z, r=1.)
        # local_Z = log_poincare(Z, centers, r=1.)
        # local_Z = local_Z.cpu().numpy()
        # print('nan values in logZ =', torch.isnan(Z).sum())
        # raise 'test'
        if not weighted:
            local_Z = Z
            kmeans = Lorentz_KMeans(
                n_clusters=n_labels,
                n_init=1,
                init="k-means++",
                max_iter=300,
                tol=1e-4,
                random_state=0,
                p=2.0,
                verbose=False,
            )
            local_labels = kmeans.fit_predict(local_Z, sample_weight=None)
            # print('kmeans labels:', local_labels.shape)
            labels_embedding = local_labels.to(device=T.device).double()
            labels_pred = (T @ labels_embedding).to(torch.int64)
        else:
            sample_weight = T.sum(0)
            idx_proto = torch.argwhere(sample_weight > 0.0)[:, 0]
            print("proto mass > 0 : ", idx_proto.shape)
            local_Z = Z[idx_proto, :]
            local_sample_weight = sample_weight[idx_proto]
            kmeans = Lorentz_KMeans(
                n_clusters=min(n_labels, idx_proto.shape[0]),
                n_init=1,
                init="k-means++",
                max_iter=300,
                tol=1e-4,
                random_state=0,
                p=2.0,
                verbose=False,
            )
            local_labels = kmeans.fit_predict(local_Z, sample_weight=local_sample_weight)
            # print('kmeans labels:', local_labels.shape)
            labels_embedding = local_labels.to(device=T.device).double()
            labels_pred = (T[:, idx_proto] @ labels_embedding).to(torch.int64)
    else:
        local_Z = Z.cpu().numpy()
        kmeans = KMeans(n_clusters=n_labels, n_init="auto")
        if weighted:
            sample_weight = T.sum(0).cpu().numpy()

        kmeans.fit(local_Z, sample_weight=sample_weight)
        # print('kmeans labels:', kmeans.labels_.shape)
        labels_embedding = torch.tensor(kmeans.labels_).to(device=T.device).double()

        labels_pred = (T @ labels_embedding).to(torch.int64)
    score = NormalizedMutualInfoScore("arithmetic")(labels_pred, Y).item()
    return score


def centroid_score(X, Z, T, Y):
    Z_ = T @ Z

    bary_X = compute_barycenters(X, Y)
    bary_Z = compute_barycenters(Z_, Y)

    # inspired from
    # https://github.com/berenslab/rna-seq-tsne/blob/master/million-cells.ipynb
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist

    dX = pdist(bary_X.cpu().numpy())
    dZ = pdist(bary_Z.cpu().numpy())

    score = spearmanr(dX[:, None], dZ[:, None]).statistic

    return score


def compute_barycenters(X, Y):
    barycenters = []
    labels = torch.unique(Y)
    delta = labels.min()
    barycenters = torch.zeros(len(labels), X.shape[-1])
    for label in labels:
        idx = Y == label
        barycenters[label - delta] = X[idx].mean(axis=0)
    return barycenters


# %%

## K-Means in the Hyperboloid model


class Lorentz_KMeans(object):
    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=0,
        p=2.0,
        lr=0.01,
        max_iter_inner=2000,
        verbose=True,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.max_iter_inner = max_iter_inner
        self.tol = tol
        self.random_state = random_state
        self.p = p
        self.lr = lr

        self.verbose = verbose
        self.labels_ = None
        self.cluster_centers_ = None

    def init_centroids(self, X, sample_weight=None):
        if self.verbose:
            print("-- init centroids --")
        random.seed(self.random_state)

        if not sample_weight is None:
            self.normalized_sample_weights = sample_weight / sample_weight.sum()
        if self.init == "random":
            centroids = X[random.sample(range(X.shape[0]), self.n_clusters), :]

        elif self.init == "k-means++":

            centroids = torch.zeros((self.n_clusters, X.shape[-1]), dtype=X.dtype, device=X.device)
            indices = set(range(X.shape[0]))

            for i in range(self.n_clusters):

                if i == 0:
                    idx = random.sample(range(X.shape[0]), self.n_clusters)[0]
                    centroids[i] = X[idx].clone()
                    indices.remove(idx)
                else:
                    l_indices = list(indices)
                    distances = (
                        torch.arccosh(
                            torch.clamp(
                                -minkowski_ip2(X[l_indices], centroids[i - 1][None, :]),
                                min=1 + 1e-15,
                            )
                        )
                        ** self.p
                    )
                    if not sample_weight is None:
                        distances = self.normalized_sample_weights[l_indices] * distances[:, 0]
                    idx = distances.argmax().item()
                    centroids[i] = X[l_indices[idx]]
                    indices.remove(l_indices[idx])

        return centroids

    def compute_centroid(self, sub_X, init_centroid, sub_sample_weight=None, verbose=False):
        """
        Compute the frechet mean using a gradient-based optimization method.
        """
        if not sub_sample_weight is None:
            norm_sub_sample_weight = sub_sample_weight / sub_sample_weight.sum()
        new_centroid = init_centroid.clone()
        new_centroid.requires_grad = True

        optimizer = RiemannianAdam([new_centroid], lr=self.lr)

        losses = []
        pbar = tqdm(range(self.max_iter_inner), desc="centroid computation", disable=not verbose)
        for i in pbar:
            optimizer.zero_grad()
            distances = (
                torch.arccosh(
                    torch.clamp(-minkowski_ip2(new_centroid[None, :], sub_X), min=1 + 1e-15)
                )
                ** self.p
            )
            if not sub_sample_weight is None:
                distances = distances * norm_sub_sample_weight[:, None]

            Loss = distances.sum()
            if torch.isnan(Loss):
                raise NanError("NaN in embedding loss")
            Loss.backward()
            optimizer.step()

            losses.append(Loss.item())
            if i > 1:
                delta = abs(losses[-1] - losses[-2]) / abs(losses[-2])
                if delta < self.tol:
                    if verbose:
                        print("---------- delta loss convergence ----------")
                    break
                if verbose:
                    pbar.set_description(
                        f"Loss : {float(losses[-1]): .3e}, " f"delta : {float(delta): .3e} "
                    )
        new_centroid.requires_grad = False
        return new_centroid.detach()

    def fit(self, X, sample_weight=None):
        n_samples = X.shape[0]

        self.inertia = None

        for run_it in range(self.n_init):
            centroids = self.init_centroids(X, sample_weight)

            for it in tqdm(range(self.max_iter), desc="fit kmeans (seed = %s)" % run_it):
                distances = (
                    torch.arccosh(torch.clamp(-minkowski_ip2(centroids, X), min=1 + 1e-15))
                    ** self.p
                )
                if not sample_weight is None:
                    distances = self.normalized_sample_weights[:, None] * distances

                # print(f'X : {X.shape} / distances : {distances.shape} / centroids: {centroids.shape}')

                labels = distances.argmin(dim=1)

                new_centroids = torch.zeros(
                    (self.n_clusters, X.shape[-1]), dtype=X.dtype, device=X.device
                )
                for i in range(self.n_clusters):
                    indices = torch.where(labels == i)[0]
                    # if self.verbose:
                    #    print('indices:', indices.shape)
                    if len(indices) > 0:
                        ### lazy computation of barycenters but too instable
                        # mean_dist = torch.mean(distances[indices, i])
                        # local_idx = torch.abs(distances[indices, i] - mean_dist).argmin().item()
                        # new_centroids[i, :] = X[indices[local_idx]]
                        if sample_weight is None:
                            new_centroids[i, :] = self.compute_centroid(
                                X[indices], centroids[i], None, verbose=False
                            )
                        else:
                            new_centroids[i, :] = self.compute_centroid(
                                X[indices],
                                centroids[i],
                                self.normalized_sample_weights[indices],
                                verbose=False,
                            )

                    else:  # handle empty cluster

                        new_centroids[i, :] = X[random.sample(range(n_samples), 1), :]

                diff_distances = 0.0
                for i in range(self.n_clusters):
                    diff_distances += (
                        torch.arccosh(
                            torch.clamp(
                                -minkowski_ip2(centroids[i, None], new_centroids[i, None]),
                                min=1 + 1e-15,
                            )
                        )
                        ** self.p
                    ).item()

                if self.verbose:
                    print(f"Seed : {run_it} / step: {it} / diff_distances: {diff_distances}")
                centroids = new_centroids.clone()
                if diff_distances < self.tol:
                    break

            distances = (
                torch.arccosh(torch.clamp(-minkowski_ip2(centroids, X), min=1 + 1e-15)) ** self.p
            )
            if not sample_weight is None:
                distances = self.normalized_sample_weights[:, None] * distances

            labels = distances.argmin(dim=1)

            inertia = 0.0
            for i in range(self.n_clusters):
                indices = torch.where(labels == i)[0]
                inertia += distances[indices, i].sum().item()

            if (self.inertia == None) or (inertia < self.inertia):
                self.inertia = inertia
                self.labels_ = labels.clone()
                self.cluster_centers_ = centroids.clone()

            if self.verbose:
                print("Iteration: {} - Best Inertia: {}".format(run_it, self.inertia))

    def fit_predict(self, X, sample_weight=None):
        self.fit(X, sample_weight)
        return self.labels_

    def fit_transform(self, X, sample_weight=None):
        self.fit(X, sample_weight)
        return self.transform(X)

    def predict(self, X, sample_weight=None):
        distances = self.transform(X, sample_weight)
        return distances.argmin(dim=1)

    def transform(self, X, sample_weight=None):
        distances = (
            torch.arccosh(torch.clamp(-minkowski_ip2(self.cluster_centers_, X), min=1 + 1e-15))
            ** self.p
        )

        if not sample_weight is None:
            normalized_sample_weights = sample_weight / sample_weight.sum()
            distances = normalized_sample_weights[:, None] * distances

        return distances


# %% macro score function


# def scores_clustdr(
#     T, Z, Y, X, threshold=0, weighted=True, hyperbolic=False, score_list=None
# ):
#     c, masses = plan_color(T, Y)

#     # retain only clusters with more than threshold points
#     ids = torch.where(T.sum(0) > threshold)[0]
#     preds = torch.argmax(T[ids], -1)

#     scores = {}

#     # torchmetrics scores
#     if "hom" in score_list:
#         scores["hom"] = HomogeneityScore()(preds, Y[ids]).item()

#     if "ami" in score_list:
#         scores["ami"] = AdjustedMutualInfoScore(average_method="arithmetic")(
#             preds, Y[ids]
#         ).item()

#     if "ari" in score_list:
#         scores["ari"] = AdjustedRandScore()(preds, Y[ids]).item()

#     if "nmi" in score_list:
#         scores["nmi"] = NormalizedMutualInfoScore("arithmetic")(preds, Y[ids]).item()

#     # kNN recall
#     if "knn_10" in score_list:
#         scores["knn_10"] = knn_recall(X, Z, T, k=10)

#     if "knn_100" in score_list:
#         scores["knn_100"] = knn_recall(X, Z, T, k=100)

#     # k-means score
#     if "kmeans" in score_list:
#         scores["kmeans"] = kmeans_score(Z, T, Y, weighted=False, hyperbolic=hyperbolic)

#     # weighted k-means score
#     if "weighted kmeans" in score_list:
#         scores["weighted kmeans"] = kmeans_score(
#             Z, T, Y, weighted=True, hyperbolic=hyperbolic
#         )

#     # centroid score
#     if "centroid" in score_list:
#         scores["centroid"] = centroid_score(X, Z, T, Y)

#     # silhouette score, if weighted=True computes weighted version
#     if "sil" in score_list:
#         if weighted:
#             pos_masses = torch.argwhere(masses > 0.0)[:, 0]
#             sub_Z, sub_c, sub_masses = Z[pos_masses], c[pos_masses], masses[pos_masses]

#             weighted_sc = weighted_silhouette_coefficients().score(
#                 sub_Z, sub_masses, sub_c, hyperbolic
#             )
#             scores["sil"] = ((weighted_sc * sub_masses).sum() / sub_masses.sum()).item()
#         else:
#             scores["sil"] = (
#                 silhouette_coefficients().score(Z[ids], c[ids], hyperbolic).sum()
#                 / ids.shape[0]
#             ).item()

#     # rounding values and multiplying by 100 for better readibility in the csv
#     scores = {key: round(100 * value, 2) for key, value in scores.items()}

#     return scores


def scores_clustdr(T, Z, Y, X, threshold=0, weighted=True, hyperbolic=False, score_list=None):
    if score_list is None:
        score_list = []

    c, masses = plan_color(T, Y)

    # Retain prototypes with more than threshold points and evaluate clustering
    # scores on the samples assigned to those active prototypes.
    active_proto_ids = torch.where(masses > threshold)[0]
    if active_proto_ids.numel() == 0:
        raise ValueError("No active prototypes remain after applying the threshold.")

    sample_proto_ids = torch.argmax(T, dim=1)
    sample_mask = masses[sample_proto_ids] > threshold
    preds = c[sample_proto_ids[sample_mask]]
    targets = Y[sample_mask]

    scores = {}

    # torchmetrics scores
    if "hom" in score_list:
        scores["hom"] = HomogeneityScore()(preds, targets).item()

    if "ami" in score_list:
        scores["ami"] = AdjustedMutualInfoScore(average_method="arithmetic")(preds, targets).item()

    if "ari" in score_list:
        scores["ari"] = AdjustedRandScore()(preds, targets).item()

    if "nmi" in score_list:
        scores["nmi"] = NormalizedMutualInfoScore("arithmetic")(preds, targets).item()

    # kNN recall
    if "knn_10" in score_list:
        scores["knn_10"] = knn_recall(X, Z, T, k=10)

    if "knn_100" in score_list:
        scores["knn_100"] = knn_recall(X, Z, T, k=100)

    # k-means score
    if "kmeans" in score_list:
        scores["kmeans"] = kmeans_score(Z, T, Y, weighted=False, hyperbolic=hyperbolic)

    # weighted k-means score
    if "weighted kmeans" in score_list:
        scores["weighted kmeans"] = kmeans_score(Z, T, Y, weighted=True, hyperbolic=hyperbolic)

    # centroid score
    if "centroid" in score_list:
        scores["centroid"] = centroid_score(X, Z, T, Y)

    # silhouette score, if weighted=True computes weighted version
    if "sil" in score_list:
        if weighted:
            pos_masses = torch.argwhere(masses > threshold)[:, 0]
            sub_Z, sub_c, sub_masses = Z[pos_masses], c[pos_masses], masses[pos_masses]

            weighted_sc = weighted_silhouette_coefficients().score(
                sub_Z, sub_masses, sub_c, hyperbolic=hyperbolic
            )
            scores["sil"] = ((weighted_sc * sub_masses).sum() / sub_masses.sum()).item()
        else:
            scores["sil"] = (
                silhouette_coefficients()
                .score(Z[active_proto_ids], c[active_proto_ids], hyperbolic=hyperbolic)
                .sum()
                / active_proto_ids.shape[0]
            ).item()

    # rounding values and multiplying by 100 for better readibility in the csv
    scores = {key: round(100 * value, 2) for key, value in scores.items()}

    return scores
