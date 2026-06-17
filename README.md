# 🧠 NanoGPT — Character-Level GPT from Scratch

> A production-grade, Decoder-Only Transformer Language Model built entirely from scratch using PyTorch — no HuggingFace, no shortcuts.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square&logo=streamlit)
![GPU](https://img.shields.io/badge/GPU-GTX%201650%204GB-76b900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Project Overview

NanoGPT is a from-scratch implementation of a **character-level Generative Pre-trained Transformer** — the same foundational architecture powering GPT-2, GPT-3, and modern LLMs. Every component is implemented manually in PyTorch without high-level abstractions, making every design decision transparent and interview-ready.

**Trained on:** [TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) — ~1MB of Shakespeare's collected works  
**Model size:** ~1.2M trainable parameters  
**Hardware target:** NVIDIA GTX 1650 (4GB VRAM)

---

## 🏗️ Architecture

```
Input (Character IDs)
       │
       ▼
┌─────────────────────────┐
│  Token Embedding        │  vocab_size → 128 dims
│  Positional Embedding   │  position  → 128 dims
└────────────┬────────────┘
             │  (sum)
             ▼
┌─────────────────────────┐  ×4 stacked blocks
│  LayerNorm              │
│  Multi-Head Attention   │  4 heads, causal mask
│  Residual Connection    │
│  LayerNorm              │
│  Feed-Forward Network   │  128 → 512 → 128
│  Residual Connection    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Final LayerNorm        │
│  LM Head (Linear)       │  128 → vocab_size
└─────────────────────────┘
             │
             ▼
       Next-Token Logits
```

| Hyperparameter | Value | Reason |
|---|---|---|
| Embedding Dim (`n_embd`) | 128 | Fits within 4GB VRAM |
| Attention Heads (`n_head`) | 4 | head_size = 32 per head |
| Transformer Layers (`n_layer`) | 4 | Balances depth vs VRAM |
| Context Window (`block_size`) | 64 | Short context, fast iteration |
| Dropout | 0.1 | Prevents overfitting on 1MB corpus |
| Batch Size | 32 | Safe for GTX 1650 |
| Learning Rate | 3e-4 | AdamW default for Transformers |

---

## 📁 Project Structure

```
gpt_project/
├── data/
│   ├── input.txt          # Auto-downloaded TinyShakespeare corpus
│   └── meta.pkl           # Serialized tokenizer (stoi / itos)
├── src/
│   ├── __init__.py
│   ├── model.py           # Full Transformer architecture from scratch
│   └── utils.py           # Tokenizer, downloader, batch sampler
├── train_script.py        # Training loop, AdamW optimizer, checkpointing
├── app.py                 # Premium Streamlit chat UI
├── model_weights.pth      # Saved model (generated after training)
├── requirements.txt       # Version-locked dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/nanogpt-from-scratch.git
cd nanogpt-from-scratch/gpt_project

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install GPU-enabled PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_script.py
```

This will:
- ✅ Auto-detect GPU (GTX 1650 or any CUDA device)  
- ✅ Download TinyShakespeare (~1MB) if not present  
- ✅ Build and save the character tokenizer to `data/meta.pkl`  
- ✅ Train for 2000 iterations (~5–10 min on GTX 1650)  
- ✅ Save model weights to `model_weights.pth`  
- ✅ Print a generated sample at the end  

**Expected output after 2000 steps:**
```
  Step    0/2000 | train loss: 4.2891 | val loss: 4.2934 | time: 0.0s
  Step  200/2000 | train loss: 2.5613 | val loss: 2.5981 | time: 30.2s
  ...
  Step 1999/2000 | train loss: 1.6842 | val loss: 1.9310 | time: 289.1s
```

### 3. Launch the Chat UI

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 💡 Key Design Decisions

### Why Causal Masking?
The model must only attend to *past and present* tokens — never future ones. A lower-triangular mask sets future attention scores to `-∞` before softmax, making their weights exactly `0`. This enforces the autoregressive constraint.

### Why Residual Connections?
In deep networks, gradients can shrink exponentially through layers (vanishing gradient problem). Residual connections create a "highway" for gradients to flow directly from the loss to early layers: `x = x + SubLayer(x)`.

### Why Pre-LayerNorm?
Normalizing inputs *before* each sub-layer (Pre-LN) is more numerically stable than the original Post-LN formulation. All modern LLMs (GPT-2+, LLaMA) use Pre-LN.

### Why AdamW over Adam?
AdamW correctly decouples weight decay from the adaptive gradient update (Loshchilov & Hutter, 2019). Standard Adam conflates L2 regularization with the adaptive moment, reducing its regularization effectiveness.

### Why Character-Level Tokenization?
- Vocabulary size ≈ 65 (vs ~50,000 for BPE) → simpler softmax
- No external tokenizer dependency
- Ideal for learning LM fundamentals; production models use BPE/SentencePiece

---

## 📊 Training Curves

| Iteration | Train Loss | Val Loss |
|-----------|------------|----------|
| 0         | ~4.29      | ~4.29    |
| 500       | ~2.12      | ~2.18    |
| 1000      | ~1.92      | ~2.01    |
| 1500      | ~1.78      | ~1.95    |
| 2000      | ~1.68      | ~1.93    |

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 4 GB (GTX 1650) | 8 GB+ |
| RAM       | 8 GB | 16 GB |
| Storage   | 500 MB | 1 GB |
| CUDA      | 11.8 | 12.x |

> **CPU Training:** Supported but ~10–20× slower. Remove `--index-url` from pip install.

---

## 🎓 Interview Explainability Guide

This project is designed to be fully explainable in technical interviews. Key concepts to discuss:

1. **Attention Mechanism** — Q/K/V projections, scaling factor `1/√d_k`, softmax
2. **Causal Masking** — Why autoregressive models can't see the future
3. **Multi-Head Attention** — Parallel attention, concatenation, output projection
4. **Positional Encoding** — Why Transformers need explicit position signals
5. **Residual Connections** — Vanishing gradient solution from He et al.
6. **Layer Normalization** — Per-token normalization across feature dimension
7. **Cross-Entropy Loss** — Log-softmax, NLL, numerical stability
8. **AdamW** — Adaptive moments + decoupled weight decay
9. **Autoregressive Decoding** — Multinomial sampling vs. greedy argmax
10. **Weight Tying** — Shared token embedding and LM head weights

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/gpt-2) — GPT-2, Radford et al., 2019
- [Let's build GPT: from scratch](https://youtu.be/kCc8FmEb1nY) — Andrej Karpathy, 2023
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2019

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
