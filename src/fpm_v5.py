"""
FriendingPredictionModelV5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the friending sequence encoder.

    This module adds positional information to the input embeddings.

    Args:
        d_model (int): The dimension of the model
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Add positional encoding to the user history sequence feature.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)  # noqa

        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        return x + self.pe[: x.size(0)]


class ComplexTimeScaling(nn.Module):
    """
    Complex time scaling module to find non-linear functions of time gaps.

    Args:
        hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to 64
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, time_gaps):
        """
        Apply complex time scaling to input time gaps.

        Args:
            time_gaps (torch.Tensor): Input tensor of time gaps

        Returns:
            torch.Tensor: Scaled time gaps
        """
        return self.mlp(time_gaps.unsqueeze(-1))


class TimeScaledAttention(nn.Module):
    """
    Time-scaled attention mechanism.

    This module extends normal self-attention by multiplying attention scores
    with a time scaling factor.

    Args:
        dim (int): Dimension of input features
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.time_scale = ComplexTimeScaling()

    def forward(self, x, time_gaps):
        """
        Apply time-scaled attention to input features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            time_gaps (torch.Tensor): Tensor of time gaps

        Returns:
            torch.Tensor: Output tensor after applying time-scaled attention
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        time_scaling = self.time_scale(time_gaps)
        attn_scores = attn_scores * time_scaling

        attn_probs = F.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


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


class FriendingPredictionModelV5(nn.Module):
    """
    Friending Prediction Model for multiple interaction probabilities in the
    friending space.

    This model predicts probabilities for various interactions such as:
    - Probability of friend request
    - Probability of friend acceptance given a friend request
    - Probability of messaging a friend (previously made)

    Args:
        user_dim (int): Dimension of user embeddings
        max_seq_len (int, optional): Maximum sequence length. Defaults to 5000.
        n_attention_layers (int, optional): Number of attention layers. Defaults to 3. # noqa
        n_tasks (int, optional): Number of tasks. Defaults to 1.
        expert_hidden_dims (list, optional): Hidden dimensions for expert networks. Defaults to [256, 128]. # noqa
        dcn_layers (int, optional): Number of DCN layers. Defaults to 3.
        num_experts (int, optional): Number of experts in MMoE. Defaults to 4.
        expert_dim (int, optional): Dimension of expert output. Defaults to 64.
    """

    def __init__(
        self,
        user_dim,
        max_seq_len=5000,
        n_attention_layers=3,
        n_tasks=1,
        expert_hidden_dims=[256, 128],
        dcn_layers=3,
        num_experts=4,
        expert_dim=64,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(user_dim, max_seq_len)

        self.attention_layers = nn.ModuleList(
            [TimeScaledAttention(user_dim) for _ in range(n_attention_layers)]
        )

        self.final_projection = nn.Sequential(
            nn.Linear(user_dim, user_dim), nn.ReLU()
        )

        # Input dimension for DCN and MMoE is 4 * user_dim
        input_dim = 4 * user_dim

        self.dcn = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(dcn_layers)]
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.mmoe = MMoE(
            input_dim, num_experts, n_tasks, expert_dim, expert_hidden_dims
        )

    def encode_sequence(self, ids, timegaps):
        """
        Encode a sequence of user interactions.

        Args:
            ids (torch.Tensor): Tensor of user IDs
            timegaps (torch.Tensor): Tensor of time gaps between interactions

        Returns:
            torch.Tensor: Encoded sequence
        """
        x = self.positional_encoding(ids)

        for attention_layer in self.attention_layers:
            x = attention_layer(x, timegaps) + x

        return x[:, -1, :]

    def forward(
        self,
        viewer_id,
        viewer_friendings,
        viewer_friending_timegap,
        target_id,
        target_friendings,
        target_friending_timegap,
    ):
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
        encoded_viewer_friendings = self.encode_sequence(
            viewer_friendings, viewer_friending_timegap
        )
        encoded_target_friendings = self.encode_sequence(
            target_friendings, target_friending_timegap
        )

        projected_viewer_friendings = self.final_projection(
            encoded_viewer_friendings
        )
        projected_target_friendings = self.final_projection(
            encoded_target_friendings
        )

        # Concatenate all features
        combined_features = torch.cat(
            [
                viewer_id,
                projected_viewer_friendings,
                target_id,
                projected_target_friendings,
            ],
            dim=1,
        )

        # Apply DCN layers
        for dcn_layer in self.dcn:
            combined_features = dcn_layer(combined_features, combined_features)

        # Apply LayerNorm
        combined_features = self.layer_norm(combined_features)

        # Apply MMoE
        task_outputs = self.mmoe(combined_features)

        return task_outputs


# Example usage
batch_size, seq_length, user_dim = 32, 10, 128
n_tasks = 3

model = FriendingPredictionModelV5(
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
