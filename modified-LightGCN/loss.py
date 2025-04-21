import torch.nn as nn
import torch
import numpy as np


class SampledSoftmaxLoss(nn.Module):
    """
    Implements the Sampled Softmax Loss function,
    acting as ArcFace Loss if a non-zero margin is specified.

    Reference: https://dl.acm.org/doi/pdf/10.1145/3637061
    """
    def __init__(self, temperature: float = 1.0, margin: float = 0.0) -> None:
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def _create_sliding_window(self, tensor, size, step=1):
        """Creates a sliding window view of the tensor."""
        return tensor.unfold(0, size, step)

    def forward(self, user_embeddings, positive_embeddings, negative_embeddings):
        """
        Computes the Sampled Softmax Loss.

        Args:
            user_embeddings (Tensor of shape (batch_size, latent_dim)):
                Embeddings of users in the batch.
            positive_embeddings (Tensor of shape (batch_size, latent_dim)):
                Embeddings of positive items in the batch.
            negative_embeddings (Tensor of shape (m * batch_size, latent_dim)):
                Embeddings of negative items, where m is the number of negative samples per user-item pair.

        Returns:
            Tensor of shape (1,), representing the computed SSM Loss.
        """

        batch_size = user_embeddings.size(0)  # number of users in batch

        negative_sample_count = negative_embeddings.size(0) // batch_size
        assert negative_embeddings.size(0) % batch_size == 0, "The number of negative samples must be a multiple of the batch size."

        norm_user_embeddings = user_embeddings / user_embeddings.norm(dim=-1, keepdim=True)
        norm_positive_embeddings = positive_embeddings / positive_embeddings.norm(dim=-1, keepdim=True)
        norm_negative_embeddings = negative_embeddings / negative_embeddings.norm(dim=-1, keepdim=True)

        positive_scores = (norm_user_embeddings * norm_positive_embeddings).sum(dim=-1) / self.temperature
        repeated_positive_scores = positive_scores.repeat_interleave(negative_sample_count)

        repeated_norm_user_embeddings = norm_user_embeddings.repeat_interleave(negative_sample_count, dim=0)
        negative_scores = (norm_negative_embeddings * repeated_norm_user_embeddings).sum(dim=-1) / self.temperature

        exponent_values = torch.exp(negative_scores - repeated_positive_scores + self.margin)
        sliding_window_sum = self._create_sliding_window(exponent_values, negative_sample_count, negative_sample_count).sum(dim=1)

        return torch.mean(torch.log1p(sliding_window_sum))