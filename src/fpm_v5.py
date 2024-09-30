"""
FriendingPredictionModelV5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """For positional encoding in the friending sequence encoder"""

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
        """Add positional encoding to the user history sequence feature"""
        return x + self.pe[: x.size(0)]


class ComplexTimeScaling(nn.Module):
    """We could encoder time with a simple nn.Linear(1, 1, bias=false)
    multiplier but this is to allow us to find more complex functions
    of timegap."""

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
        return self.mlp(time_gaps.unsqueeze(-1))


class TimeScaledAttention(nn.Module):
    """Only addition to normal self attention is to multiply with
    a score coming from the TimeScaling module. This will independently
    learn recency."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.time_scale = ComplexTimeScaling()

    def forward(self, x, time_gaps):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        time_scaling = self.time_scale(time_gaps)
        attn_scores = attn_scores * time_scaling

        attn_probs = F.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_probs, v)


class CrossLayer(nn.Module):
    """Used in DCN

    DCN layers are called with (combined_features, combined_features)
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        x = (
            x0 * (torch.sum(x * self.weight, dim=-1, keepdim=True) + self.bias)
            + x
        )
        return x


class Expert(nn.Module):
    """For MMoE"""

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MMoE(nn.Module):
    """Multi gated mixture of experts"""

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
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(
            expert_outputs, dim=1
        )  # [batch_size, num_experts, expert_dim]

        final_outputs = []
        for task in range(self.num_tasks):
            gate_output = F.softmax(
                self.gates[task](x), dim=1
            )  # [batch_size, num_experts]
            gated_output = torch.sum(
                gate_output.unsqueeze(-1) * expert_outputs, dim=1
            )  # [batch_size, expert_dim]
            final_outputs.append(self.task_specific_layers[task](gated_output))

        return final_outputs


class FriendingPredictionModel(nn.Module):
    """
    This is a common model that is tasked with predicting the probability of
    multiple interactions in the friending space. For instance:
    - probability of friend request
    - probability of friend acceptance | friend request
    - probability of messaging a friend (previously made)
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

        self.final_projection = nn.Linear(user_dim, user_dim)

        # Input dimension for DCN and MMoE is 4 * user_dim
        input_dim = 4 * user_dim

        self.dcn = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(dcn_layers)]
        )
        self.mmoe = MMoE(
            input_dim, num_experts, n_tasks, expert_dim, expert_hidden_dims
        )

    def encode_sequence(self, ids, timegaps):
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

        # Apply MMoE
        task_outputs = self.mmoe(combined_features)

        return task_outputs


# Example usage
batch_size, seq_length, user_dim = 32, 10, 128
n_tasks = 3

model = FriendingPredictionModel(
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
