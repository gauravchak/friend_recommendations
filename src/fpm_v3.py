"""
FriendingPredictionModelV3
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    """
    Cross layer used in Deep & Cross Network (DCN).

    Args:
        input_dim (int): Dimension of input features
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        """
        Apply cross layer operation.

        Args:
            x0 (torch.Tensor): Initial input tensor
            x (torch.Tensor): Current input tensor

        Returns:
            torch.Tensor: Output tensor after applying cross layer operation
        """
        x = (
            x0 * (torch.sum(x * self.weight, dim=-1, keepdim=True) + self.bias)
            + x
        )
        return x


class Expert(nn.Module):
    """
    Expert network for Multi-gate Mixture-of-Experts (MMoE).

    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        hidden_dims (list): List of hidden dimensions for the expert network
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply expert network to input features.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying expert network
        """
        return self.net(x)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) module.

    Args:
        input_dim (int): Dimension of input features
        num_experts (int): Number of experts
        num_tasks (int): Number of tasks
        expert_dim (int): Dimension of expert output
        hidden_dims (list): List of hidden dimensions for expert networks
    """

    def __init__(
        self, input_dim, num_experts, num_tasks, expert_dim, hidden_dims
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.experts = nn.ModuleList(
            [
                Expert(input_dim, expert_dim, hidden_dims)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, num_experts) for _ in range(num_tasks)]
        )
        self.task_specific_layers = nn.ModuleList(
            [nn.Linear(expert_dim, 1) for _ in range(num_tasks)]
        )

    def forward(self, x):
        """
        Apply MMoE to input features.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            list: List of output tensors for each task
        """
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(
            expert_outputs, dim=1
        )  # [batch_size, num_experts, expert_dim]

        final_outputs = []
        for task in range(self.num_tasks):
            gate_output = F.softmax(
                self.gates[task](x) + 1e-10, dim=1
            )  # [batch_size, num_experts]
            gated_output = torch.sum(
                gate_output.unsqueeze(-1) * expert_outputs, dim=1
            )  # [batch_size, expert_dim]
            final_outputs.append(self.task_specific_layers[task](gated_output))

        return final_outputs


class FriendingPredictionModelV3(nn.Module):
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
        expert_hidden_dims (list, optional): Hidden dimensions for expert networks. Defaults to [256, 128]. # noqa
        dcn_layers (int, optional): Number of DCN layers. Defaults to 3.
        num_experts (int, optional): Number of experts in MMoE. Defaults to 4.
        expert_dim (int, optional): Dimension of expert output. Defaults to 64.
    """

    def __init__(
        self,
        user_dim: int,
        n_tasks: int,
        expert_hidden_dims: List[int] = [256, 128],
        dcn_layers: int = 3,
        num_experts: int = 4,
        expert_dim: int = 64,
    ):
        super().__init__()
        # Input dimension for DCN and MMoE is 4 * user_dim
        input_dim = 4 * user_dim

        self.dcn = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(dcn_layers)]
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.mmoe = MMoE(
            input_dim, num_experts, n_tasks, expert_dim, expert_hidden_dims
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
        # We are assuming that you are transforming viewer_id,
        # viewer_friendings, target_id, target_friendings through embedding
        # tables and padding to produce [B, D], [B, N, D], [B, D], [B, N, D]

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

        # Apply DCN layers
        for dcn_layer in self.dcn:
            # DCN + skip connection
            combined_features = combined_features + dcn_layer(
                combined_features, combined_features
            )

        # Apply LayerNorm
        combined_features = self.layer_norm(combined_features)

        # Apply MMoE
        task_logits = self.mmoe(combined_features)
        return task_logits


# Example usage
batch_size, seq_length, user_dim = 32, 100, 128
n_tasks = 3

model = FriendingPredictionModelV3(
    user_dim=user_dim, n_tasks=n_tasks, dcn_layers=3, num_experts=4
)

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
