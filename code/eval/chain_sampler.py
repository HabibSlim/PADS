"""
Chain sampler for shape chains.
"""
from torch.utils.data.sampler import Sampler


class ChainSampler(Sampler):
    def __init__(self, dataset, batch_size, chain_length):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chain_length = chain_length
        self.num_samples = len(dataset)
        self.n_batches = self.num_samples // (self.batch_size)
        self.n_chains = self.num_samples // self.chain_length

    def __iter__(self):
        t = 0
        for m in range(self.n_chains):
            for k in range(self.chain_length):
                for j in range(self.batch_size):
                    yield (
                        m * (self.chain_length * self.batch_size)
                        + k
                        + j * self.chain_length
                    )
                    t += 1
                    if t == self.num_samples:
                        return

    def __len__(self):
        return self.num_samples
