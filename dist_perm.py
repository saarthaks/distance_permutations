import torch
import numpy as np

class DistPerm:
    def __init__(self, k, d=None):
        self.k = k

        if d and d <= k:
            self.d = d
        else:
            self.d = k

        self.anchors = k*[None]
        self.index = []
        self.index_inv = []
        self.is_trained = False

    def fit(self, training_data, alg='random'):
        if alg == 'random':
            self.anchors = training_data[np.random.choice(len(training_data),
                                                          size=self.k,
                                                          replace=False)]
            self.is_trained = True
            return self.anchors
        else:
            return NotImplementedError

    def add(self, database):
        assert(self.is_trained)

        distances = torch.cdist(database, self.anchors, p=2)
        self.index = torch.argsort(distances, dim=-1)[:, :self.d]
        self.index_inv = torch.argsort(self.index, dim=-1).float()
        # print(self.index_inv.dtype)

        return self.index, self.index_inv

    def search(self, query, num):
        assert(self.is_trained)

        q_dist = torch.cdist(query, self.anchors, p=2)
        query_perm = torch.argsort(q_dist, dim=-1)[:, :self.d]
        query_perm_inv = torch.argsort(query_perm, dim=-1).float()
        db_dists = torch.cdist(self.index_inv, query_perm_inv, p=2)
        closest_idx = torch.topk(db_dists, num, dim=0, largest=False)[1].transpose(0,1)

        return closest_idx
