# =============================================================================
# src/model.py — Character-Level GPT: Transformer Decoder Architecture
# =============================================================================
#
# Implements a Decoder-Only Transformer (GPT-style) from scratch in PyTorch.
#
# Architecture Overview:
#   Input IDs → Token Embedding + Positional Embedding
#             → N × [LayerNorm → Multi-Head Self-Attention → LayerNorm → FFN]
#             → Final LayerNorm
#             → Linear Head (projects to vocab_size logits)
#             → Softmax → Next-Token Probabilities
#
# Key Design Principles:
#   • Pre-norm architecture (LayerNorm before each sub-layer) — more stable
#     than the original "post-norm" Transformer (Vaswani et al., 2017).
#   • Residual (skip) connections prevent vanishing gradients in deep stacks.
#   • Causal masking ensures the model cannot attend to future tokens, which
#     is required for autoregressive (left-to-right) language modeling.
#   • Learned positional embeddings (vs. sinusoidal) — simpler and equally
#     effective for the sequence lengths used here.
#
# Resume-Optimized Hyperparameters (fits comfortably in 4GB VRAM):
#   n_embd   = 128  (embedding / hidden dimension)
#   n_head   = 4    (attention heads per layer)
#   n_layer  = 4    (number of stacked Transformer Blocks)
#   block_size = 64 (maximum context window in tokens)
#   dropout  = 0.1  (regularization — prevents memorizing training data)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Resume-Optimized Default Hyperparameters
# ---------------------------------------------------------------------------
# These values are chosen so the full model fits within 4GB VRAM (GTX 1650).
# Scaling n_embd / n_layer / n_head would increase capacity at a higher cost.

DEFAULT_BLOCK_SIZE = 64    # Context window: how many tokens the model "sees"
DEFAULT_N_EMBD     = 128   # Hidden size: width of all internal representations
DEFAULT_N_HEAD     = 4     # Number of parallel attention heads
DEFAULT_N_LAYER    = 4     # Depth: number of stacked Transformer Blocks
DEFAULT_DROPOUT    = 0.1   # Dropout rate applied to attention & FFN outputs


# =============================================================================
# Component 1: Scaled Dot-Product Self-Attention Head
# =============================================================================

class Head(nn.Module):
    """
    Single causal self-attention head.

    Core Idea (Attention is All You Need, Vaswani et al. 2017):
      Each token computes three vectors from its embedding:
        • Query  (Q): "What information am I looking for?"
        • Key    (K): "What information do I contain?"
        • Value  (V): "What information will I share?"

      Attention scores = softmax(Q @ Kᵀ / √d_k) @ V

      The scaling factor (1/√d_k) prevents dot products from growing too
      large in magnitude, which would push softmax into saturation regions
      with near-zero gradients — destabilizing training.

    Causal Masking:
      A lower-triangular boolean mask (tril) sets all future-token positions
      to -∞ before softmax, driving their attention weights to 0. This enforces
      the autoregressive constraint: position t can only attend to positions ≤ t.
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()

        # Linear projections — no bias, consistent with GPT-2 convention
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: lower-triangular matrix of shape (block_size, block_size)
        # Registered as a buffer so it moves to GPU with .to(device) but is
        # not a learnable parameter (excluded from optimizer updates).
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape (B, T, C) where
                        B = batch size, T = sequence length, C = embedding dim.
        Returns:
            Tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape

        # Project inputs to Q, K, V spaces
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Compute raw attention scores with scaling
        # Scaling by 1/√d_k keeps variance stable regardless of head_size
        scale = self.head_size ** -0.5
        att   = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Apply causal mask: mask out upper triangle (future positions)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax normalizes scores into a proper probability distribution
        att = F.softmax(att, dim=-1)                 # (B, T, T)
        att = self.dropout(att)                      # Regularization

        # Weighted aggregation of values
        out = att @ v   # (B, T, head_size)
        return out


# =============================================================================
# Component 2: Multi-Head Self-Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multiple parallel attention heads concatenated and projected.

    Why multiple heads?
      Different heads can learn to attend to different aspects of context
      simultaneously. For example, one head may focus on syntactic structure
      while another captures semantic relationships — this is the "multi-head"
      intuition from Vaswani et al.

    The outputs of all heads are concatenated along the feature dimension
    (restoring dimensionality to n_embd) and then linearly projected to allow
    cross-head information mixing.
    """

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()

        # head_size × n_head = n_embd ensures the concat output stays at n_embd
        head_size = n_embd // n_head

        # Parallel heads — in practice can be fused into one batched operation
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)
        ])

        # Output projection mixes information across all head outputs
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate all head outputs along the last (feature) dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)

        # Project back to n_embd and apply dropout
        out = self.dropout(self.proj(out))
        return out


# =============================================================================
# Component 3: Position-wise Feed-Forward Network (FFN)
# =============================================================================

class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently at each sequence position.

    Role in the Transformer:
      After attention aggregates context *across* positions, the FFN
      processes each position *independently*, allowing the model to apply
      nonlinear transformations to the attended representations.

      The 4× expansion (n_embd → 4*n_embd → n_embd) follows the original
      Transformer paper and gives the FFN extra representational capacity.

    Activation: ReLU is used here for simplicity; GELU is used in GPT-2/3.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # Expand: project to 4× hidden dim
            nn.ReLU(),                         # Nonlinearity — breaks linearity
            nn.Linear(4 * n_embd, n_embd),    # Contract: project back to n_embd
            nn.Dropout(dropout),               # Regularize to prevent overfitting
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Component 4: Transformer Block
# =============================================================================

class Block(nn.Module):
    """
    Single Transformer decoder layer — the fundamental building block.

    Pre-Norm Architecture:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

      LayerNorm is applied *before* each sub-layer (pre-norm), which is
      more numerically stable than the original post-norm formulation.
      This is used by GPT-2 and all modern large language models.

    Residual (Skip) Connections:
      The addition (x + ...) allows gradients to flow directly from the
      output loss to early layers without passing through all the sub-layers.
      This solves the vanishing gradient problem, enabling deeper networks.
      Without residuals, training >4 layers reliably would be very difficult.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()

        # Self-attention sub-layer
        self.sa   = MultiHeadAttention(n_head, n_embd, block_size, dropout)

        # Feed-forward sub-layer
        self.ffwd = FeedForward(n_embd, dropout)

        # Layer normalization applied before each sub-layer (Pre-LN)
        # LayerNorm normalizes across the feature dimension per token,
        # decoupling the scale from the gradient flow.
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize → attend → residual add
        x = x + self.sa(self.ln1(x))

        # Pre-LN: normalize → feed-forward → residual add
        x = x + self.ffwd(self.ln2(x))

        return x


# =============================================================================
# Component 5: NanoGPT — Top-Level Model
# =============================================================================

class NanoGPT(nn.Module):
    """
    Decoder-Only Transformer Language Model (GPT-style).

    Full forward pass:
      1. Token Embedding   : Map integer token IDs → dense vectors (n_embd dims)
      2. Position Embedding: Add learned positional signal to each token vector
      3. Transformer Blocks: n_layer stacked Block layers refine representations
      4. Final LayerNorm   : Stabilize activations before the language model head
      5. LM Head           : Linear layer projecting n_embd → vocab_size logits

    Why learned positional embeddings?
      Unlike the original sinusoidal encodings, learned embeddings are trained
      end-to-end and can capture arbitrary positional patterns. They work well
      for fixed context lengths and are simpler to implement.

    Hyperparameters tuned for GTX 1650 (4GB VRAM):
      Roughly 1.2M parameters — small enough to train comfortably on 4GB.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd:     int   = DEFAULT_N_EMBD,
        n_head:     int   = DEFAULT_N_HEAD,
        n_layer:    int   = DEFAULT_N_LAYER,
        block_size: int   = DEFAULT_BLOCK_SIZE,
        dropout:    float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.block_size = block_size

        # --- Embedding Tables ---
        # Token embedding: each of the vocab_size token IDs → n_embd vector
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)

        # Position embedding: each of the block_size positions → n_embd vector
        # The model learns which positional patterns matter for this corpus.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # --- Transformer Stack ---
        # n_layer Blocks applied sequentially, each refining the representations
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # --- Final Normalization ---
        # Applied after all blocks, before the LM head — standard in GPT-2+
        self.ln_f = nn.LayerNorm(n_embd)

        # --- Language Model Head ---
        # Projects hidden states to unnormalized vocabulary logits
        # No softmax here — cross-entropy loss applies it internally for
        # numerical stability (log-sum-exp trick).
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Weight tying: share weights between token embedding and LM head
        # This regularization technique (Press & Wolf, 2016) reduces parameters
        # and often improves perplexity on small datasets.
        self.token_embedding_table.weight = self.lm_head.weight

        # Initialize weights using a small normal distribution
        # This is important: random init at the right scale avoids exploding
        # activations before the first training step.
        self.apply(self._init_weights)

        print(f"[model] NanoGPT initialized | Parameters: {self.count_params():,}")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Custom weight initialization.
        Linear and Embedding layers use N(0, 0.02) — consistent with GPT-2.
        Biases are zeroed out.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the full model.

        Args:
            idx     (Tensor): Integer token IDs, shape (B, T).
            targets (Tensor): Ground-truth next tokens, shape (B, T). Optional —
                              if None, the model runs in inference mode (no loss).

        Returns:
            logits (Tensor):      Raw logits of shape (B, T, vocab_size).
            loss   (Tensor|None): Scalar cross-entropy loss, or None in inference.
        """
        B, T = idx.shape

        # Safety check: context longer than block_size is unsupported
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds model block_size {self.block_size}."
        )

        # --- Token + Position Embeddings ---
        tok_emb = self.token_embedding_table(idx)           # (B, T, n_embd)

        # Positional embedding — arange gives [0, 1, ..., T-1] for each batch
        pos     = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)        # (T, n_embd)

        # Add positional signal to token embeddings (broadcasting over B)
        x = tok_emb + pos_emb                               # (B, T, n_embd)

        # --- Transformer Blocks ---
        x = self.blocks(x)                                  # (B, T, n_embd)

        # --- Final Layer Norm ---
        x = self.ln_f(x)                                    # (B, T, n_embd)

        # --- Language Model Head ---
        logits = self.lm_head(x)                            # (B, T, vocab_size)

        # --- Loss Computation ---
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            # Cross-entropy internally applies log-softmax → stable training
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressive text generation using greedy sampling.

        At each step:
          1. Crop the context to the last block_size tokens (sliding window)
          2. Forward pass to get logits for the *last* position
          3. Convert logits to probabilities via softmax
          4. Sample the next token from the distribution (multinomial)
          5. Append the sampled token and repeat

        Using multinomial sampling (vs. argmax) produces more varied,
        creative text — argmax would greedily repeat the most likely token.

        Args:
            idx            (Tensor): Seed context of shape (B, T).
            max_new_tokens (int):    Number of tokens to generate.

        Returns:
            Tensor of shape (B, T + max_new_tokens) with generated tokens appended.
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (model's memory limit)
            idx_cond = idx[:, -self.block_size:]

            # Forward pass — only logits needed, so targets=None
            logits, _ = self(idx_cond)

            # Focus on the logits at the last time step (next-token prediction)
            logits = logits[:, -1, :]              # (B, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)      # (B, vocab_size)

            # Sample one token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)

        return idx
