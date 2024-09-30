"""
FriendingPredictionModelV2
"""

import torch
import torch.nn as nn


def compute_pairwise_interactions(
    combined_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise elementwise multiplication for all pairs of features.

    Args:
        combined_features (torch.Tensor): Input tensor of shape [B, K, D]

    Returns:
        torch.Tensor: Tensor of pairwise interactions with shape [B, KC2, D]
    """
    batch_size, num_features, feature_dim = combined_features.shape

    # Create all possible pairs of indices
    pairs = torch.combinations(torch.arange(num_features), r=2)
    num_pairs: int = len(pairs)

    # Initialize the output tensor
    pairwise_interactions = torch.zeros(
        batch_size, num_pairs, feature_dim, device=combined_features.device
    )

    # Compute elementwise multiplication for each pair
    for i, (idx1, idx2) in enumerate(pairs):
        pairwise_interactions[:, i, :] = (
            combined_features[:, idx1, :] * combined_features[:, idx2, :]
        )

    return pairwise_interactions


class FriendingPredictionModelV2(nn.Module):
    """
    Friending Prediction Model for multiple interaction probabilities in the
    friending space.

    This model predicts probabilities for various interactions such as:
    - Probability of friend request
    - Probability of friend acceptance given a friend request
    - Probability of messaging a friend (previously made)

    Args:
        user_dim (int): Dimension of user embeddings
        n_tasks (int, optional): Number of tasks.
    """

    def __init__(
        self,
        user_dim: int,
        n_tasks: int,
    ):
        super().__init__()
        # 10 = 4C2 + 4
        task_arch_hidden_dim: int = 512
        self.task_arch = nn.Sequential(
            nn.Linear(
                in_features=10 * user_dim, out_features=task_arch_hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout layer for regularization
            nn.Linear(task_arch_hidden_dim, n_tasks),
        )

    def forward(
        self,
        viewer_id: torch.Tensor,
        viewer_friendings: torch.Tensor,
        viewer_friending_timegap: torch.Tensor,
        target_id: torch.Tensor,
        target_friendings: torch.Tensor,
        target_friending_timegap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Friending Prediction Model.

        Args:
            viewer_id (torch.Tensor): Tensor of viewer IDs
            viewer_friendings (torch.Tensor): Tensor of viewer friending sequences  # noqa
            viewer_friending_timegap (torch.Tensor): Tensor of viewer friending time gaps  # noqa
            target_id (torch.Tensor): Tensor of target IDs
            target_friendings (torch.Tensor): Tensor of target friending sequences  # noqa
            target_friending_timegap (torch.Tensor): Tensor of target friending time gaps  # noqa

        Returns:
            list: List of output tensors for each task
        """
        mean_pooled_viewer_friendings = torch.mean(
            viewer_friendings, dim=1
        )  # mean pool viewer_friendings [B, N, D] to [B, D]
        mean_pooled_target_friendings = torch.mean(
            target_friendings, dim=1
        )  # mean pool viewer_friendings [B, N, D] to [B, D]
        # Stack all features
        combined_features = torch.cat(
            [
                viewer_id,
                mean_pooled_viewer_friendings,
                target_id,
                mean_pooled_target_friendings,
            ],
            dim=1,
        )
        pairwise_interactions = compute_pairwise_interactions(combined_features)
        all_features = torch.cat(
            (combined_features, pairwise_interactions), dim=1
        )
        # flatten all_features and pass through task arch
        batch_size, _, _ = all_features.shape
        task_logits = self.task_arch(all_features.reshape(batch_size, -1))
        return task_logits


# Example usage
batch_size, seq_length, user_dim = 32, 10, 128
n_tasks = 3

model = FriendingPredictionModelV2(user_dim=user_dim, n_tasks=n_tasks)

# Generate dummy data
viewer_id = torch.randn(batch_size, user_dim)
viewer_friendings = torch.randn(batch_size, seq_length, user_dim)
viewer_friending_timegap = torch.randn(batch_size, seq_length)
target_id = torch.randn(batch_size, user_dim)
target_friendings = torch.randn(batch_size, seq_length, user_dim)
target_friending_timegap = torch.randn(batch_size, seq_length)

# Forward pass
outputs = model(
    viewer_id,
    viewer_friendings,
    viewer_friending_timegap,
    target_id,
    target_friendings,
    target_friending_timegap,
)

for i, tensor in enumerate(outputs):
    print(f"Task {i+1} output shape:", tensor.shape)
