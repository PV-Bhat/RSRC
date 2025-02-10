# file: code_snippets/moe_sparsity/moe_layer_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SparseMoELayer(nn.Module):
    """
    A simplified implementation of a sparse Mixture-of-Experts (MoE) layer in PyTorch.
    
    This layer routes each input sample to the top-k experts based on a gating network.
    For top-k=1 (default), it selects the single expert with the highest gating score.
    For top-k>1, it computes a weighted sum of the top-k experts' outputs.
    
    Args:
        num_experts (int): Number of expert networks.
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of output features.
        top_k (int): Number of top experts to use per sample. Default is 1.
    """
    def __init__(self, num_experts: int, input_dim: int, output_dim: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts as linear layers
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        # Gating network to compute expert selection probabilities
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        batch_size = x.size(0)
        # Compute gating scores and probabilities
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        gate_probs = torch.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # Compute expert outputs: list of [batch_size, output_dim]
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack into tensor: [batch_size, num_experts, output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        if self.top_k == 1:
            # Top-1 gating: select the expert with the highest probability for each input
            top_expert_indices = torch.argmax(gate_probs, dim=-1)  # [batch_size]
            # Use advanced indexing to select the expert outputs
            final_output = expert_outputs[torch.arange(batch_size), top_expert_indices]
        else:
            # Top-k gating: compute a weighted sum of the top-k expert outputs
            topk_probs, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)  # [batch_size, top_k]
            # Normalize top-k probabilities
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [batch_size, top_k]
            # Gather top-k expert outputs and compute weighted sum
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))  # [batch_size, top_k, output_dim]
            topk_expert_outputs = torch.gather(expert_outputs, 1, topk_indices_exp)  # [batch_size, top_k, output_dim]
            final_output = (topk_expert_outputs * topk_probs.unsqueeze(-1)).sum(dim=1)  # [batch_size, output_dim]
        
        return final_output

if __name__ == '__main__':
    # Example usage with top-1 gating
    moe_layer = SparseMoELayer(num_experts=4, input_dim=128, output_dim=256, top_k=1)
    input_tensor = torch.randn(32, 128)
    output = moe_layer(input_tensor)
    print("MoE Layer Output Shape (top-1):", output.shape)  # Expected: torch.Size([32, 256])
    
    # Example usage with top-2 gating (weighted combination)
    moe_layer_top2 = SparseMoELayer(num_experts=4, input_dim=128, output_dim=256, top_k=2)
    output_top2 = moe_layer_top2(input_tensor)
    print("MoE Layer Output Shape (top-2):", output_top2.shape)  # Expected: torch.Size([32, 256])
