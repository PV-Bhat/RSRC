# file: code_snippets/recursive_regularization/entropy_penalized_quantization_example.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution.
    
    Args:
        probs (torch.Tensor): A tensor representing a probability distribution.
    
    Returns:
        torch.Tensor: A scalar tensor representing the entropy.
    """
    return -torch.sum(probs * torch.log(probs + 1e-9))

def uniform_quantize(weights: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly quantize the weights to a fixed number of bits.
    
    Args:
        weights (torch.Tensor): The weights to quantize.
        num_bits (int): The number of bits for quantization.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - quantized_weights: The quantized weights.
            - q_indices: Quantization indices corresponding to the quantized weights.
    """
    num_levels = 2 ** num_bits
    w_min = weights.min()
    w_max = weights.max()
    scale = (w_max - w_min) / (num_levels - 1) if w_max != w_min else 1.0
    # Quantize: map weights to nearest level indices
    q_indices = torch.round((weights - w_min) / scale)
    quantized_weights = q_indices * scale + w_min
    return quantized_weights, q_indices

def compute_quantization_entropy(q_indices: torch.Tensor, num_levels: int) -> torch.Tensor:
    """
    Compute the entropy of the quantized weights by constructing a histogram.
    
    Args:
        q_indices (torch.Tensor): The quantization indices.
        num_levels (int): The total number of quantization levels.
    
    Returns:
        torch.Tensor: A scalar tensor representing the entropy.
    """
    # Flatten the indices and compute histogram counts (non-differentiable, conceptual)
    hist = torch.histc(q_indices.float(), bins=num_levels, min=0, max=num_levels - 1)
    # Convert counts to probabilities
    probs = hist / (hist.sum() + 1e-9)
    return entropy(probs)

def entropy_penalized_quantization(weights: torch.Tensor, num_bits: int = 8, lambda_entropy: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply uniform quantization with an added entropy penalty.
    
    Args:
        weights (torch.Tensor): The weights to quantize.
        num_bits (int): Number of bits for quantization.
        lambda_entropy (float): Weight for the entropy penalty term.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - quantized_weights: The quantized weights.
            - total_loss: The combined loss (quantization error plus entropy penalty).
    """
    num_levels = 2 ** num_bits
    # Uniform quantization
    quantized_weights, q_indices = uniform_quantize(weights, num_bits)
    # Compute quantization error (Mean Squared Error)
    quant_error = F.mse_loss(quantized_weights, weights)
    # Compute entropy of the quantized weight distribution
    q_entropy = compute_quantization_entropy(q_indices, num_levels)
    # Total loss combines the quantization error with the entropy penalty
    total_loss = quant_error + lambda_entropy * q_entropy
    return quantized_weights, total_loss

if __name__ == '__main__':
    # Example usage: apply entropy penalized quantization on a linear layer's weights
    linear_layer = nn.Linear(128, 256)
    original_weights = linear_layer.weight.data.clone()
    
    quantized_weights, loss = entropy_penalized_quantization(original_weights, num_bits=8, lambda_entropy=0.01)
    
    print("Original Weights (first 5 values):", original_weights.flatten()[:5])
    print("Quantized Weights (first 5 values):", quantized_weights.flatten()[:5])
    print("Entropy Penalized Quantization Loss:", loss.item())
