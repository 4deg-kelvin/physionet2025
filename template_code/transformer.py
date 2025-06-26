import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# --- Helper Modules ---

class ResidualBlock(nn.Module):
    """
    Implements the residual block as described in Figure 2.
    The paper describes two types of residual blocks. This class handles both.
    
    If project_shortcut is True:
        y = F(x) + BN(Conv1x1(x))  (Residual Network 1 in the diagram)
    Else:
        y = F(x) + x               (Residual Network 2 in the diagram)
    
    F(x) = BN(Conv1x1(ReLU(BN(Conv1x1(x)))))
    """
    def __init__(self, d_model: int, project_shortcut: bool = False):
        """
        Args:
            d_model (int): The feature dimension of the input.
            project_shortcut (bool): If True, use a 1x1 conv for the shortcut connection.
        """
        super().__init__()
        self.project_shortcut = project_shortcut

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(d_model)

        if self.project_shortcut:
            self.shortcut_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.shortcut_bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Conv1d expects (batch_size, channels, length)
        # so we permute (B, L, D) to (B, D, L)
        x_permuted = x.permute(0, 2, 1)

        # Shortcut connection
        if self.project_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x_permuted))
        else:
            shortcut = x_permuted

        # Main path
        out = self.conv1(x_permuted)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut
        out += shortcut
        
        # Permute back to (B, L, D)
        return out.permute(0, 2, 1)


class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network from the Transformer architecture.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --- Core Cardioformer Modules ---

class MultiGranularityPatchEmbedding(nn.Module):
    """
    Handles step (1) from the paper: Cross-Channel Multi-Granularity Patch Embedding.
    It takes a raw time series and converts it into multiple sets of patch embeddings,
    one for each granularity level.
    """
    def __init__(self, seq_len: int, patch_lengths: List[int], in_channels: int, d_model: int):
        super().__init__()
        self.patch_lengths = patch_lengths
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_granularities = len(patch_lengths)
        self.seq_len = seq_len
        
        # Create a linear projection layer for each granularity
        self.projection_layers = nn.ModuleList([
            nn.Linear(L * in_channels, d_model) for L in patch_lengths
        ])

        # Learnable embeddings for each granularity level
        self.granularity_embeddings = nn.Parameter(torch.randn(self.num_granularities, 1, d_model))

        # --- Create a single, fixed positional encoding table ---
        # The table needs to be large enough for the granularity with the most patches.
        max_patches = 0
        if self.patch_lengths:
            max_patches = self.seq_len // min(self.patch_lengths)
        
        # We create a buffer so it's part of the model's state and moves to the correct device.
        self.register_buffer('positional_encoding', self._get_positional_encoding(max_patches + 1, self.d_model))


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input ECG signal of shape (batch_size, seq_len, in_channels).
        
        Returns:
            A tuple containing:
            - A list of patch embeddings for each granularity.
            - A list of router embeddings for each granularity.
        """
        batch_size = x.shape[0]
        # Assert that the input sequence length matches the configured one.
        # Permute the second to third dim
        assert x.shape[1] == self.seq_len, f"Input sequence length ({x.shape[1]}) must match model's configured seq_len ({self.seq_len})"

        patch_embeddings_list = []
        
        for i, L in enumerate(self.patch_lengths):
            # 1. Calculate the number of non-overlapping patches.
            num_patches = self.seq_len // L
            
            # 2. Trim the sequence to be perfectly divisible by the patch length.
            effective_len = num_patches * L
            x_slice = x[:, :effective_len, :]

            # 3. Reshape the slice into patches.
            # Shape: (batch_size, num_patches, patch_length, in_channels)
            patches = x_slice.reshape(batch_size, num_patches, L, self.in_channels)
            
            # 4. Flatten patches for the linear layer.
            # Shape: (batch_size, num_patches, patch_length * in_channels)
            patches_flat = patches.flatten(2)
            
            # 5. Project flattened patches to d_model.
            patch_embed = self.projection_layers[i](patches_flat) # Shape: (batch_size, num_patches, d_model)
            
            # 6. Add positional and granularity embeddings. This is the critical step.
            # `patch_embed` has shape (B, N, D).
            # `self.positional_encoding[:num_patches, :]` has shape (N, D).
            # `self.granularity_embeddings[i]` has shape (1, D).
            # Broadcasting handles the addition correctly.
            final_embed = patch_embed + self.positional_encoding[:num_patches, :] + self.granularity_embeddings[i]
            
            patch_embeddings_list.append(final_embed)

        # 7. Create router embeddings using the same positional encoding table.
        router_embeddings_list = [
            (self.positional_encoding[self.seq_len // L, :] + self.granularity_embeddings[i]).expand(batch_size, 1, -1)
            for i, L in enumerate(self.patch_lengths)
        ]

        return patch_embeddings_list, router_embeddings_list
    
    def _get_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class MultiGranularityAttention(nn.Module):
    """
    Implements step (2) from the paper: Multi-Granularity Self-Attention.
    This involves a two-stage process: Intra-granularity and Inter-granularity attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Attention modules
        self.intra_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
    def forward(self, patch_embeds: List[torch.Tensor], router_embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            patch_embeds (List[torch.Tensor]): List of patch embeddings from MultiGranularityPatchEmbedding.
            router_embeds (List[torch.Tensor]): List of router embeddings from MultiGranularityPatchEmbedding.
        
        Returns:
            List[torch.Tensor]: List of updated patch embeddings after both attention stages.
        """
        updated_patch_embeds = []
        updated_router_embeds = []

        # --- Stage 1: Intra-Granularity Self-Attention ---
        for x_i, u_i in zip(patch_embeds, router_embeds):
            # z = [x || u]
            z_i = torch.cat([x_i, u_i], dim=1) # (B, N_i + 1, D)
            
            # Refine patch embeddings: x_i' = Attn(q=x_i, k=z_i, v=z_i)
            x_i_updated, _ = self.intra_attn(query=x_i, key=z_i, value=z_i, need_weights=False)
            updated_patch_embeds.append(x_i_updated)
            
            # Refine router embeddings: u_i' = Attn(q=u_i, k=z_i, v=z_i)
            u_i_updated, _ = self.intra_attn(query=u_i, key=z_i, value=z_i, need_weights=False)
            updated_router_embeds.append(u_i_updated)

        # --- Stage 2: Inter-Granularity Self-Attention ---
        # U = [u_1' || u_2' || ... || u_n']
        U = torch.cat(updated_router_embeds, dim=1) # (B, num_granularities, D)
        
        final_router_embeds = []
        for u_i in updated_router_embeds:
            # Update each router again using cross-granularity context
            # u_i'' = Attn(q=u_i, k=U, v=U)
            u_i_final, _ = self.inter_attn(query=u_i, key=U, value=U, need_weights=False)
            final_router_embeds.append(u_i_final)
        
        # The paper is slightly ambiguous on how the final routers update the patch embeddings.
        # A reasonable interpretation is that the updated patches from the intra-attention
        # are the final output of this block, and the final routers are discarded or used
        # implicitly in the next layer's attention. We will return the updated_patch_embeds.
        return updated_patch_embeds


class CardioformerEncoderLayer(nn.Module):
    """
    A single layer of the Cardioformer encoder, as depicted in Figure 2.
    This deviates from the standard Transformer encoder.
    Flow: Attention -> ResNet1 -> ResNet2 (x2) -> FeedForward -> LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiGranularityAttention(d_model, n_heads, dropout)
        
        # According to the paper, there are 3 residual blocks.
        # The first is different from the other two.
        self.res_net1 = ResidualBlock(d_model, project_shortcut=True)
        self.res_net2 = ResidualBlock(d_model, project_shortcut=False)
        self.res_net3 = ResidualBlock(d_model, project_shortcut=False)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, patch_embeds: List[torch.Tensor], router_embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            patch_embeds (List[torch.Tensor]): List of patch embeddings.
            router_embeds (List[torch.Tensor]): List of router embeddings.
        
        Returns:
            List[torch.Tensor]: List of transformed patch embeddings.
        """
        # --- Attention ---
        # The attention block doesn't have a residual connection or norm before it in the diagram
        attn_outputs = self.attention(patch_embeds, router_embeds)
        
        # Process each granularity's output through the rest of the layer
        final_outputs = []
        for i, x in enumerate(attn_outputs):
            # --- Residual Networks ---
            # Diagram shows skip connections around each block
            x_res1 = self.res_net1(x)
            x = self.norm1(x + x_res1)
            
            x_res2 = self.res_net2(x)
            x = self.norm2(x + x_res2)
            
            x_res3 = self.res_net3(x)
            x = self.norm3(x + x_res3)

            # --- Feed Forward ---
            x_ffn = self.ffn(x)
            x = self.norm4(x + x_ffn)
            
            final_outputs.append(x)
            
        return final_outputs

# --- Main Model ---

class Cardioformer(nn.Module):
    """
    The complete Cardioformer model for ECG classification.
    """
    def __init__(self, seq_len: int, in_channels: int,
                 patch_lengths: List[int], d_model: int = 128, n_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. Embedding Layer
        self.patch_embedding = MultiGranularityPatchEmbedding(self.seq_len, patch_lengths, in_channels, d_model)
        
        # 2. Encoder Layers
        self.layers = nn.ModuleList([
            CardioformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # 3. Projection Head
        # The paper states: "These [patch embeddings] are concatenated to form the final
        # representation h, which is used to predict the label y".
        num_patches_total = sum(seq_len // L for L in patch_lengths)
        self.projection = nn.Linear(num_patches_total * d_model, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input ECG data of shape (batch_size, in_channels, seq_len).
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, 1).
        """
        # 1. Get initial multi-granularity patch and router embeddings
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, in_channels)
        patch_embeds, router_embeds = self.patch_embedding(x)
        
        # 2. Pass through Cardioformer encoder layers
        # Note: The routers are regenerated by the embedding layer and used inside each attention layer.
        # They are not passed from one encoder layer to the next.
        for layer in self.layers:
            patch_embeds = layer(patch_embeds, router_embeds)
            
        # 3. Concatenate all patch embeddings from all granularities
        # Flatten the sequence dimension for each granularity's patches
        h = torch.cat(patch_embeds, dim=1) # (batch_size, total_num_patches, d_model)
        h_flat = h.flatten(start_dim=1) # (batch_size, total_num_patches * d_model)

        # 4. Final classification
        logits = self.projection(self.dropout(h_flat))

        return logits

# if __name__ == '__main__':
#     # --- Model Configuration for Chagas Disease (Binary Classification) ---
#     SEQ_LEN = 250
#     IN_CHANNELS = 12  # Standard 12-lead ECG
#     NUM_CLASSES = 1   # Binary classification (e.g., Chagas vs. No Chagas)
#                       # The output will be a single logit.
    
#     # Using a simpler patch list for demonstration
#     # The paper uses a long, specific list: {2, 4, 8, 8, 16, 16, ..., 32}
#     PATCH_LENGTHS = [4, 8, 16, 32] 
    
#     D_MODEL = 128 # Embedding dimension
#     D_FF = 256    # Feed-forward hidden dimension
#     N_HEADS = 8   # Number of attention heads
#     NUM_LAYERS = 6 # Number of encoder layers
#     DROPOUT = 0.1
    
#     BATCH_SIZE = 16

#     # --- Instantiate the Model ---
#     print("--- Initializing Cardioformer Model for Binary Classification ---")
#     model = Cardioformer(
#         seq_len=SEQ_LEN,
#         in_channels=IN_CHANNELS,
#         patch_lengths=PATCH_LENGTHS,
#         d_model=D_MODEL,
#         n_heads=N_HEADS,
#         num_layers=NUM_LAYERS,
#         d_ff=D_FF,
#         dropout=DROPOUT
#     )
    
#     print("\n--- Model Architecture ---")
#     print(model)
    
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nTotal trainable parameters: {num_params:,}")

#     # --- Create a Dummy Input ---
#     dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
#     print(f"\n--- Running a Forward Pass ---")
#     print(f"Input shape: {dummy_input.shape}")

#     # --- Get Model Output ---
#     try:
#         output_logits = model(dummy_input)
#         print(f"Output logits shape: {output_logits.shape}")
#         assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        
#         # For binary classification, you pass the logits through a sigmoid
#         # to get probabilities.
#         output_probs = torch.sigmoid(output_logits)
#         print(f"Output probabilities shape: {output_probs.shape}")
        
#         print("\n✅ Forward pass successful and output shape is correct for binary classification!")

#     except Exception as e:
#         print(f"\n❌ An error occurred during the forward pass: {e}")