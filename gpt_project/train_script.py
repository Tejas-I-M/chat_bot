# =============================================================================
# train_script.py — Training Infrastructure for Character-Level NanoGPT
# =============================================================================
#
# Responsibilities:
#   1. Hardware auto-detection (CUDA GPU vs CPU fallback)
#   2. Data acquisition and tokenization via src/utils.py
#   3. Model instantiation with resume-grade hyperparameters
#   4. Training loop with periodic loss evaluation (train + val)
#   5. Checkpoint saving to 'model_weights.pth'
#
# Hardware Target: NVIDIA GTX 1650 (4GB VRAM)
#   • Batch size = 32  (conservative to stay within 4GB)
#   • Block size = 64  (short context window)
#   • n_embd = 128     (small embedding dimension)
#   → Estimated peak VRAM usage: ~350MB — well within GTX 1650 limits.
#
# Estimated Runtime:
#   ~5–10 minutes on GTX 1650 for 2000 iterations over TinyShakespeare.
# =============================================================================

import os
import sys
import time
import torch
from typing import Dict

# Make sure src/ is importable regardless of where this script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import (
    ensure_data_exists,
    build_tokenizer,
    save_tokenizer,
    prepare_datasets,
    get_batch,
)
from model import NanoGPT


# =============================================================================
# Hyperparameters
# =============================================================================
# All hyperparameters are grouped here at the top of the file — a convention
# that makes experiments easy to track and reproduce (no hunting through code).

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---
TRAIN_FRAC = 0.9     # 90% training, 10% validation split

# --- Model Architecture (GTX 1650 VRAM-safe) ---
BLOCK_SIZE = 64      # Sequence context window (tokens)
N_EMBD     = 128     # Embedding / hidden dimension
N_HEAD     = 4       # Number of attention heads per block
N_LAYER    = 4       # Number of stacked Transformer blocks
DROPOUT    = 0.1     # Dropout probability (regularization)

# --- Training ---
BATCH_SIZE    = 32        # Sequences per gradient update
MAX_ITERS     = 2000      # Total training iterations
LEARNING_RATE = 3e-4      # AdamW learning rate (standard for small Transformers)
EVAL_INTERVAL = 200       # Evaluate train/val loss every N iterations
EVAL_ITERS    = 100       # Number of batches averaged during evaluation

# --- Output ---
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.pth")


# =============================================================================
# Hardware Verification
# =============================================================================

def verify_hardware() -> None:
    """
    Logs hardware information and confirms GPU availability.

    For GTX 1650 users: training on CUDA is ~10–20× faster than CPU.
    If CUDA is unavailable, the script falls back to CPU automatically.
    """
    print("=" * 60)
    print("  NanoGPT Training Script")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device selected : {DEVICE.upper()}")

    if DEVICE == "cuda":
        gpu_name  = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU             : {gpu_name}")
        print(f"  Total VRAM      : {total_mem:.1f} GB")
    else:
        print("  ⚠ CUDA not available. Training on CPU (will be slow).")

    print("=" * 60)
    print()


# =============================================================================
# Loss Evaluation
# =============================================================================

@torch.no_grad()
def estimate_loss(
    model:      NanoGPT,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
) -> Dict[str, float]:
    """
    Computes average loss over multiple random batches from train & val sets.

    Decorated with @torch.no_grad() to:
      • Disable gradient computation entirely (no autograd graph built)
      • Significantly reduce memory usage during evaluation
      • Speed up inference since we only need forward passes here

    We set model.eval() to disable dropout during evaluation — dropout is a
    training-time regularization technique and should be inactive at eval time.
    After evaluation, model.train() re-enables dropout for the next iteration.

    Args:
        model      (NanoGPT): The model being trained.
        train_data (Tensor):  Encoded training corpus.
        val_data   (Tensor):  Encoded validation corpus.

    Returns:
        dict with 'train' and 'val' float loss values.
    """
    losses = {}
    model.eval()   # Disable dropout for clean evaluation

    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split_name] = split_losses.mean().item()

    model.train()  # Re-enable dropout for training
    return losses


# =============================================================================
# Main Training Loop
# =============================================================================

def train() -> None:
    """
    Full training pipeline: data → model → optimize → checkpoint.

    Training Loop Design:
      AdamW (Adaptive Moment Estimation with Weight Decay) is used because:
        • Adam adapts learning rates per-parameter → faster convergence than SGD
        • Weight decay (L2 regularization) decoupled from gradient update
          (Loshchilov & Hutter, 2019) — prevents overfitting on small datasets
        • lr=3e-4 is the empirically best default for small Transformer models

      Cross-Entropy Loss:
        The model outputs logits (unnormalized log-probabilities). PyTorch's
        cross_entropy internally computes log-softmax + NLL loss in a single
        numerically stable operation, avoiding overflow from large exponentials.
    """
    verify_hardware()

    # -------------------------------------------------------------------------
    # Step 1: Data Acquisition and Tokenization
    # -------------------------------------------------------------------------
    print("[train] Step 1/4: Loading and tokenizing corpus ...")
    text = ensure_data_exists()

    stoi, itos, vocab_size = build_tokenizer(text)
    save_tokenizer(stoi, itos, vocab_size)

    train_data, val_data = prepare_datasets(text, stoi, TRAIN_FRAC)
    print()

    # -------------------------------------------------------------------------
    # Step 2: Model Instantiation
    # -------------------------------------------------------------------------
    print("[train] Step 2/4: Instantiating NanoGPT model ...")
    model = NanoGPT(
        vocab_size = vocab_size,
        n_embd     = N_EMBD,
        n_head     = N_HEAD,
        n_layer    = N_LAYER,
        block_size = BLOCK_SIZE,
        dropout    = DROPOUT,
    ).to(DEVICE)

    # Log model summary
    total_params = model.count_params()
    print(f"[train] Total trainable parameters: {total_params:,}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Optimizer Setup
    # -------------------------------------------------------------------------
    print("[train] Step 3/4: Setting up AdamW optimizer ...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 1e-2,   # L2 regularization — decoupled from Adam's moments
        betas        = (0.9, 0.95),  # Standard Transformer Adam betas
    )
    print(f"[train] Optimizer    : AdamW | lr={LEARNING_RATE} | wd=1e-2")
    print(f"[train] Batch size   : {BATCH_SIZE}")
    print(f"[train] Max iters    : {MAX_ITERS}")
    print(f"[train] Eval every   : {EVAL_INTERVAL} steps")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Training Loop
    # -------------------------------------------------------------------------
    print("[train] Step 4/4: Starting training loop ...")
    print("-" * 60)

    start_time = time.time()

    for step in range(MAX_ITERS):

        # Periodic evaluation — measures generalization (train vs val gap)
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start_time
            print(
                f"  Step {step:>4d}/{MAX_ITERS} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"time: {elapsed:.1f}s"
            )

        # --- Forward Pass ---
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        logits, loss = model(x, y)

        # --- Backward Pass ---
        # Zero gradients first — PyTorch accumulates gradients by default,
        # so we must clear them before each new backward pass.
        optimizer.zero_grad(set_to_none=True)   # set_to_none=True is faster
                                                 # than zero_grad() alone

        # Compute gradients via backpropagation (autograd)
        loss.backward()

        # Gradient clipping: cap gradient norms at 1.0 to prevent
        # exploding gradients — common in Transformer training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters using the computed gradients
        optimizer.step()

    # -------------------------------------------------------------------------
    # Checkpoint: Save Model Weights
    # -------------------------------------------------------------------------
    print("-" * 60)
    print(f"\n[train] Training complete in {time.time() - start_time:.1f}s")
    print(f"[train] Saving model state dict to '{WEIGHTS_PATH}' ...")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[train] [OK] Checkpoint saved successfully.")

    # -------------------------------------------------------------------------
    # Quick Sanity Check: Generate a sample
    # -------------------------------------------------------------------------
    print("\n[train] Generating a quick sample (128 tokens) ...")
    from utils import decode

    model.eval()
    seed = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # Start token: 0
    with torch.no_grad():
        generated_ids = model.generate(seed, max_new_tokens=128)[0].tolist()
    sample_text = decode(generated_ids, itos)

    print("-" * 60)
    print(sample_text)
    print("-" * 60)
    print("\n[train] Done. Run 'streamlit run app.py' to launch the chat UI.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    train()
