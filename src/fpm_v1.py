"""
FriendingPredictionModelV1
"""

import torch
import torch.nn as nn


class FriendingPredictionModelV1(nn.Module):
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
        self.task_arch = nn.Sequential(
            nn.Linear(in_features=4 * user_dim, out_features=n_tasks), nn.ReLU()
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
        # Concatenate all features
        combined_features = torch.cat(
            [
                viewer_id,
                mean_pooled_viewer_friendings,
                target_id,
                mean_pooled_target_friendings,
            ],
            dim=1,
        )
        task_logits = self.task_arch(combined_features)
        return task_logits


# Example usage
batch_size, seq_length, user_dim = 32, 10, 128
n_tasks = 3

model = FriendingPredictionModelV1(user_dim=user_dim, n_tasks=n_tasks)

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
