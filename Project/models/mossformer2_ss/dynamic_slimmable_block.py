import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

class DynamicSlimmableBlock(nn.Module):
    """
    Dynamic Slimmable Block (DSB) for adaptive computational complexity in speech separation,
    inspired by Elminshawi et al. (IEEE Signal Processing Letters, 2024).
    Replaces static FFN with a slimmable sub-network and gating module to adjust width based
    on input chunk characteristics (e.g., temporal mean and std).
    """
    def __init__(self, in_channels: int, out_channels: int, lorder: int = 20,
                 utilization_factors: List[float] = [0.125, 1.0], hidden_size: int = 1024,
                 tau_init: float = 3.0, alpha: float = 0.1, beta: float = 0.01):
        """
        Args:
            in_channels (int): Dimension of the input features.
            out_channels (int): Dimension of the output features.
            lorder (int): Length of the order for convolution layers (default: 20).
            utilization_factors (List[float]): Pre-defined utilization factors (e.g., [0.125, 1.0]).
            hidden_size (int): Number of hidden units in the linear layer (default: 1024).
            tau_init (float): Initial sharpness parameter for softmax in gating (default: 3.0).
            alpha (float): Weighting factor for gate sparsity loss (default: 0.1).
            beta (float): Weighting factor for importance loss (default: 0.01).
        """
        super(DynamicSlimmableBlock, self).__init__()

        # Validate input parameters
        if not in_channels > 0 or not out_channels > 0:
            raise ValueError("Input and output channels must be positive.")
        if not all(0 < u <= 1 for u in utilization_factors):
            raise ValueError("Utilization factors must be in (0, 1].")
        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lorder = lorder
        self.utilization_factors = sorted(utilization_factors, reverse=True)
        self.hidden_size = hidden_size
        self.num_factors = len(utilization_factors)
        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.alpha = alpha
        self.beta = beta

        # Slimmable linear layers (Eq. 1 logic)
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_channels, bias=False)
        self.conv = nn.Conv2d(out_channels, out_channels, [lorder + lorder - 1, 1],
                              [1, 1], groups=out_channels, bias=False)

        # Gating module (Eq. 2)
        self.gate = nn.Linear(2 * in_channels, self.num_factors)
        self.register_buffer('importance', torch.zeros(self.num_factors))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with dynamic slimming based on input chunk stats.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, time, channels].

        Returns:
            Tuple[torch.Tensor, dict]: Gated output and loss dictionary.
        """
        batch_size, time_steps, _ = x.shape

        # Input validation
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(f"Expected input shape [batch, time, {self.in_channels}], got {x.shape}")

        # Chunk-level stats (mean, std) as per Eq. (2)
        chunk_mean = x.mean(dim=1, keepdim=True)  # [batch, 1, channels]
        chunk_std = x.std(dim=1, keepdim=True)    # [batch, 1, channels]
        gate_input = torch.cat([chunk_mean, chunk_std], dim=-1)  # [batch, 1, 2*channels]
        gate_input = gate_input.squeeze(1)  # [batch, 2*channels]

        # Gating module: Predict utilization probabilities
        logits = self.tau * self.gate(gate_input)  # [batch, num_factors]
        probs = F.softmax(logits, dim=-1)  # [batch, num_factors]

        # Update importance buffer for loss computation
        self.importance.data = torch.sum(probs, dim=0) / batch_size

        # Apply soft gating during training (Eq. 3) or hard gating during inference (Eq. 4)
        if self.training:
            out = torch.zeros(batch_size, time_steps, self.out_channels, device=x.device)
            for k, u in enumerate(self.utilization_factors):
                slimmed_x = self._slimmable_forward(x, u)
                out += probs[:, k:k+1] * slimmed_x  # Weighted sum over all utilizations
        else:
            k_star = probs.argmax(dim=-1)  # [batch]
            out = torch.stack([self._slimmable_forward(x[i:i+1], self.utilization_factors[k])
                              for i, k in enumerate(k_star)], dim=0)

        # Compute auxiliary losses (Eqs. 6, 8)
        loss_dict = self._compute_losses(probs)

        return out, loss_dict

    def _slimmable_forward(self, x: torch.Tensor, u: float) -> torch.Tensor:
        """
        Compute forward pass with slimmed weights for a given utilization factor.

        Args:
            x (torch.Tensor): Input tensor [batch, time, channels].
            u (float): Utilization factor in (0, 1].

        Returns:
            torch.Tensor: Slimmed output.
        """
        batch_size, time_steps, _ = x.shape
        hidden_dim = int(u * self.hidden_size)

        # Slim linear1 weights (Eq. 1)
        slimmed_w1 = self.linear1.weight[:hidden_dim, :]
        slimmed_linear1 = nn.Linear(self.in_channels, hidden_dim, bias=True)
        slimmed_linear1.weight.data = slimmed_w1
        slimmed_linear1.bias.data = self.linear1.bias[:hidden_dim]

        # Slim linear2 weights
        slimmed_w2 = self.linear2.weight[:, :hidden_dim]
        slimmed_linear2 = nn.Linear(hidden_dim, self.out_channels, bias=False)
        slimmed_linear2.weight.data = slimmed_w2

        # Forward pass with slimmed layers
        f1 = F.relu(slimmed_linear1(x))  # [batch, time, hidden_dim]
        p1 = slimmed_linear2(f1)  # [batch, time, out_channels]
        x_conv = p1.unsqueeze(1)  # [batch, 1, time, out_channels]
        x_per = x_conv.permute(0, 3, 2, 1)  # [batch, out_channels, time, 1]
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])  # Causal padding
        conv_out = x_per + self.conv(y)  # [batch, out_channels, time, 1]
        out = conv_out.permute(0, 2, 1, 3).squeeze(-1)  # [batch, time, out_channels]

        return out

    def _compute_losses(self, probs: torch.Tensor) -> dict:
        """
        Compute auxiliary losses for training (Eqs. 6, 8).

        Args:
            probs (torch.Tensor): Probability tensor [batch, num_factors].

        Returns:
            dict: Dictionary containing sparsity and importance losses.
        """
        batch_size = probs.size(0)
        norm_probs = probs / (torch.norm(probs, p=2, dim=-1, keepdim=True) + 1e-8)
        sparsity_loss = torch.mean(torch.sum(self.alpha * torch.abs(norm_probs), dim=-1))

        importance = self.importance
        cv = torch.std(importance) / (torch.mean(importance) + 1e-8)
        importance_loss = self.beta * cv ** 2

        return {'sparsity_loss': sparsity_loss, 'importance_loss': importance_loss}

    def extra_repr(self) -> str:
        """
        Provide a string representation of the module.
        """
        return (f"DynamicSlimmableBlock(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, lorder={self.lorder}, "
                f"hidden_size={self.hidden_size}, num_factors={self.num_factors}, "
                f"utilization_factors={self.utilization_factors})")

# Example usage (for testing)
if __name__ == "__main__":
    # Initialize the block
    dsb = DynamicSlimmableBlock(in_channels=256, out_channels=256, lorder=20,
                                utilization_factors=[0.125, 0.5, 1.0], hidden_size=1024)

    # Generate random input
    x = torch.randn(4, 100, 256)  # [batch, time, channels]
    output, losses = dsb(x)

    # Print shapes and losses
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sparsity loss: {losses['sparsity_loss'].item():.4f}")
    print(f"Importance loss: {losses['importance_loss'].item():.4f}")

    # Verify gradient flow
    output.sum().backward()
    print("Gradient check passed if no errors.")