import torch
from torch.utils.data import Dataset, DataLoader
import random


# Function to generate a sequence with varying probabilities for 1s
def generate_sequence(length, p1, p2, special_regions):
    sequence = torch.zeros(length)
    global_prob = torch.rand(length)
    sequence[global_prob < p1] = 1

    for region in special_regions:
        a, b = region
        if 0 <= a < b <= length:
            special_prob = torch.rand(b - a)
            sequence[a:b][special_prob < p2] = 1

    return sequence


# Custom Dataset class
class SequenceDataset(Dataset):
    def __init__(self, seq_len, p1, p2, special_regions, num_samples):
        """
        Initializes the dataset.

        Args:
        - seq_len: Length of each sequence.
        - p1: Global probability for 1s.
        - p2: Probability for 1s in special regions.
        - special_regions: List of regions with higher probability.
        - num_samples: Number of sequences in the dataset.
        """
        self.seq_len = seq_len
        self.p1 = p1
        self.p2 = p2
        self.special_regions = special_regions
        self.num_samples = num_samples

    def __len__(self):
        # Total number of samples in the dataset
        return self.num_samples

    def __getitem__(self, idx):
        # Generates a sequence
        if random.random() < 0.5:
            sequence = generate_sequence(self.seq_len, self.p1, self.p1, self.special_regions)
            return sequence, 0
        else:
            sequence = generate_sequence(self.seq_len, self.p1, self.p2, self.special_regions)
            return sequence, 1


if __name__ == '__main__':
    # Create dataset and dataloader
    seq_len = 100  # Length of each sequence
    p1 = 0.1  # Global probability for 1s
    p2 = 0.5  # Probability for 1s in special regions
    special_regions = [(10, 200), (50, 60)]  # Regions with higher probability
    num_samples = 50  # Number of sequences to generate

    # Create the dataset and the dataloader
    dataset = SequenceDataset(seq_len, p1, p2, special_regions, num_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example of iterating through the DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(batch)
        if i == 2:  # Just to display the first 3 batches
            break
