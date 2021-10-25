import torch
import h5py
import numpy as np

class SubspaceDataset:
    def __init__(self, d, D):
        self.d = d
        self.D = D

        self.means = 0.5 + 0.09*np.random.randn(d)
        self.Cov = 0.01*np.eye(d)
        self.L = (np.random.randn(d, D) + 1)/d

        self.database = None

    def generate_db(self, num_samples):
        assert(self.database is None)

        data_low = np.random.multivariate_normal(self.means, self.Cov,
                                                 size=num_samples)
        self.database_low = torch.from_numpy(data_low).float()
        data = data_low @ self.L
        self.database = torch.from_numpy(data).float()

        return self.database

    def generate_queries(self, num_queries):
        queries_low = np.random.multivariate_normal(self.means, self.Cov,
                                                    size=num_queries)
        queries = queries_low @ self.L

        return torch.from_numpy(queries).float(), torch.from_numpy(queries_low).float()

class SIFTDataset:
    def __init__(self, filepath, ntot):
        self.file = h5py.File(filepath, 'r')
        self.ntot = ntot

    def generate_db(self):
        data = np.array(self.file.get('fea'))
        samples = data[:self.ntot :]
        # samples -= samples.mean(axis=0)

        return torch.from_numpy(samples).float()

    def generate_queries(self, num_queries):
        data = np.array(self.file.get('fea'))
        queries = data[np.random.choice(self.ntot, size=num_queries), :]

        return torch.from_numpy(queries).float()
