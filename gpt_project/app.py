# =============================================================================
# app.py — Premium ChatGPT-Inspired Web UI (Streamlit)
# =============================================================================
#
# A production-quality chat interface for the NanoGPT character-level language
# model. Features a dark-themed workspace, persistent chat history, technical
# sidebar, and model-cached inference for a seamless user experience.
#
# Run with: streamlit run app.py
# =============================================================================

import os
import sys
import torch
import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import NanoGPT, DEFAULT_BLOCK_SIZE, DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_DROPOUT
from utils import load_tokenizer, encode, decode

# ---------------------------------------------------------------------------
# Page Configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title    = "NanoGPT Chat",
    page_icon     = "🧠",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# Custom CSS — Dark-Themed Premium UI
# =============================================================================

st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Reset & Background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0f14 0%, #111827 50%, #0d1117 100%);
        color: #e6edf3;
        min-height: 100vh;
    }

    /* ── Hide Streamlit Default Decorations ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0 !important;
        max-width: 900px;
    }

    /* ── App Header / Hero ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #a371f7 50%, #ff7b72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    .header-badge {
        display: inline-block;
        background: rgba(88, 166, 255, 0.15);
        border: 1px solid rgba(88, 166, 255, 0.3);
        color: #58a6ff;
        border-radius: 999px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }

    /* ── Chat Container ── */
    .chat-container {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        min-height: 420px;
        max-height: 520px;
        overflow-y: auto;
        scroll-behavior: smooth;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.6rem;
        padding: 0.2rem;
        animation: fadeSlideIn 0.25s ease-out both;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0);   }
    }

    /* User message bubble */
    [data-testid="stChatMessage"][data-role="user"] {
        background: rgba(88, 166, 255, 0.08);
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    /* Assistant message bubble */
    [data-testid="stChatMessage"][data-role="assistant"] {
        background: rgba(163, 113, 247, 0.07);
        border: 1px solid rgba(163, 113, 247, 0.18);
    }

    /* ── Chat Input Box ── */
    [data-testid="stChatInput"] {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid rgba(48, 54, 61, 0.9) !important;
        border-radius: 12px !important;
        color: #e6edf3 !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(88, 166, 255, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.12) !important;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(48, 54, 61, 0.6) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #58a6ff;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── Metric Cards (sidebar specs) ── */
    .spec-card {
        background: rgba(88, 166, 255, 0.06);
        border: 1px solid rgba(88, 166, 255, 0.18);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    .spec-card:hover {
        background: rgba(88, 166, 255, 0.1);
        border-color: rgba(88, 166, 255, 0.35);
    }
    .spec-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .spec-value {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .spec-unit {
        font-size: 0.75rem;
        color: #58a6ff;
        margin-left: 0.3rem;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(35, 134, 54, 0.15);
        border: 1px solid rgba(35, 134, 54, 0.35);
        color: #3fb950;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #3fb950;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    /* ── Warning Pill (model not loaded) ── */
    .warning-pill {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid rgba(210, 153, 34, 0.35);
        color: #e3b341;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* ── Generation Settings Slider ── */
    .stSlider > div > div > div { background: #58a6ff !important; }

    /* ── Divider ── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(48, 54, 61, 0.7);
        margin: 1rem 0;
    }

    /* ── Clear Chat Button ── */
    .stButton > button {
        background: rgba(248, 81, 73, 0.1) !important;
        border: 1px solid rgba(248, 81, 73, 0.3) !important;
        color: #f85149 !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: rgba(248, 81, 73, 0.2) !important;
        border-color: rgba(248, 81, 73, 0.5) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(88, 166, 255, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(88, 166, 255, 0.5); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Constants — must match train_script.py
# =============================================================================

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.pth")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Cached Model Loader
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Loads the trained NanoGPT model and character tokenizer once into memory.

    @st.cache_resource ensures this function runs exactly once per session,
    preventing redundant GPU memory allocations on each Streamlit rerender.
    Streamlit caches the returned objects (model, stoi, itos) for the lifetime
    of the app process.

    Returns:
        model (NanoGPT): Trained model in eval mode on the target device.
        stoi  (dict):    Character → integer encoding map.
        itos  (dict):    Integer → character decoding map.

    Raises:
        RuntimeError if model_weights.pth or data/meta.pkl are missing.
    """
    stoi, itos, vocab_size = load_tokenizer()

    model = NanoGPT(
        vocab_size = vocab_size,
        n_embd     = DEFAULT_N_EMBD,
        n_head     = DEFAULT_N_HEAD,
        n_layer    = DEFAULT_N_LAYER,
        block_size = DEFAULT_BLOCK_SIZE,
        dropout    = DEFAULT_DROPOUT,
    )

    # Load saved weights — map_location ensures CPU fallback works too
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()   # Disable dropout for inference

    return model, stoi, itos


# =============================================================================
# Sidebar — Model Architecture Specs
# =============================================================================

def render_sidebar(model_loaded: bool) -> int:
    """Renders the expandable technical sidebar with architecture info."""

    with st.sidebar:
        # Header
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem">
            <div style="font-size:2rem">🧠</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e6edf3;">NanoGPT</div>
            <div style="font-size:0.78rem; color:#8b949e;">Character-Level LLM</div>
        </div>
        """, unsafe_allow_html=True)

        # Model status indicator
        if model_loaded:
            st.markdown("""
            <div class="status-pill">
                <div class="status-dot"></div> Model Loaded · GPU Ready
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-pill">
                ⚠ Model not loaded.<br>Run <code>train_script.py</code> first.
            </div>
            """, unsafe_allow_html=True)

        # ── Architecture Specs Expander ──
        with st.expander("⚙ Model Architecture Specs", expanded=True):

            def spec_card(label, value, unit=""):
                st.markdown(f"""
                <div class="spec-card">
                    <div class="spec-label">{label}</div>
                    <div class="spec-value">{value}<span class="spec-unit">{unit}</span></div>
                </div>
                """, unsafe_allow_html=True)

            spec_card("Architecture",    "Decoder-Only Transformer")
            spec_card("Embedding Dim",   DEFAULT_N_EMBD,  "d_model")
            spec_card("Attention Heads", DEFAULT_N_HEAD,  "heads")
            spec_card("Transformer Layers", DEFAULT_N_LAYER, "blocks")
            spec_card("Context Window",  DEFAULT_BLOCK_SIZE, "tokens")
            spec_card("Dropout",         DEFAULT_DROPOUT, "rate")
            spec_card("Tokenizer",       "Character-Level")
            spec_card("Training Data",   "TinyShakespeare", "~1MB")
            spec_card("Optimizer",       "AdamW (lr=3e-4)")
            spec_card("Hardware Target", "NVIDIA GTX 1650")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Generation Settings ──
        st.markdown("## 🎛 Generation Settings")
        max_tokens = st.slider(
            "Max New Tokens",
            min_value = 50,
            max_value = 500,
            value     = 200,
            step      = 25,
            help      = "Number of characters the model generates per response.",
        )

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Device Info ──
        device_icon = "⚡" if DEVICE == "cuda" else "🖥"
        device_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
        st.markdown(f"""
        <div class="spec-card">
            <div class="spec-label">Active Device</div>
            <div class="spec-value">{device_icon} {device_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Clear Chat ──
        if st.button("🗑 Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    return max_tokens


# =============================================================================
# App Header
# =============================================================================

def render_header() -> None:
    st.markdown("""
    <div class="app-header">
        <h1>NanoGPT Chat</h1>
        <p>A Character-Level Generative Transformer built from scratch with PyTorch</p>
        <span class="header-badge">DECODER-ONLY TRANSFORMER · GTX 1650 OPTIMIZED</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat(model, stoi: dict, itos: dict, max_tokens: int) -> None:
    """
    Renders the persistent chat window and handles user input.

    Session State Design:
      st.session_state.messages stores the full conversation history as a list
      of {"role": ..., "content": ...} dicts. Streamlit's session_state persists
      across rerenders triggered by widget interactions, simulating a stateful app.
    """

    # Initialize chat history on first load
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role"   : "assistant",
                "content": (
                    "Hello! I am **NanoGPT** — a Decoder-Only Transformer trained "
                    "on Shakespeare's collected works. Give me a seed prompt and I'll "
                    "continue writing in the spirit of the Bard. 🎭\n\n"
                    "*Try: \"HAMLET:\", \"To be or not\", \"KING RICHARD:\"*"
                ),
            }
        ]

    # Render all existing messages from session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])

    # ── Input Box ──
    if prompt := st.chat_input(
        placeholder="Enter a seed prompt (e.g. 'To be or not to be') ...",
        key="chat_input",
    ):
        # Append and display the user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # ── Generation ──
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Generating ..."):
                try:
                    # Encode the user's prompt to integer token IDs
                    context_ids = encode(prompt, stoi)

                    # Clamp context to block_size — model has a fixed memory limit
                    context_ids = context_ids[-DEFAULT_BLOCK_SIZE:]
                    idx = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)

                    # Autoregressive generation — the model predicts one token at a
                    # time and feeds it back in, extending the sequence step by step.
                    with torch.no_grad():
                        output_ids = model.generate(idx, max_new_tokens=max_tokens)

                    # Decode the full output (prompt + generated tokens)
                    full_ids    = output_ids[0].tolist()
                    output_text = decode(full_ids, itos)

                    # Extract only the newly generated portion (after the prompt)
                    generated = output_text[len(prompt):]

                    # Format the response
                    response = f"**[Seed]** `{prompt}`\n\n**[Generated]**\n\n{generated}"
                    st.markdown(response)

                    # Save assistant response to session state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except KeyError as e:
                    error_msg = (
                        f"⚠ **Character not in vocabulary:** `{e}`\n\n"
                        "The model was trained on Shakespeare's text and only knows "
                        "characters from that corpus. Please use standard ASCII text."
                    )
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

                except Exception as e:
                    error_msg = f"⚠ **Generation error:** {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


# =============================================================================
# Main Application Entry Point
# =============================================================================

def main() -> None:
    """Orchestrates the Streamlit app layout and component rendering."""

    render_header()

    # ── Try to load the model ──
    model_loaded = False
    model        = None
    stoi         = None
    itos         = None

    if os.path.exists(WEIGHTS_PATH):
        try:
            model, stoi, itos = load_model_and_tokenizer()
            model_loaded = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    # else: model_loaded remains False

    # ── Render Sidebar (always visible) ──
    max_tokens = render_sidebar(model_loaded)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Panel ──
    if model_loaded:
        render_chat(model, stoi, itos, max_tokens)
    else:
        # Friendly onboarding message when model is not yet trained
        st.markdown("""
        <div style="
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid rgba(48, 54, 61, 0.8);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            backdrop-filter: blur(12px);
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
            <h3 style="color: #e6edf3; margin-bottom: 0.5rem;">Model Not Trained Yet</h3>
            <p style="color: #8b949e; max-width: 480px; margin: 0 auto 1.5rem;">
                To get started, run the training script to download the dataset,
                train NanoGPT, and save the model weights.
            </p>
            <div style="
                background: rgba(22, 27, 34, 0.9);
                border: 1px solid rgba(48, 54, 61, 0.8);
                border-radius: 10px;
                padding: 1rem 1.5rem;
                display: inline-block;
                text-align: left;
            ">
                <code style="color: #58a6ff; font-size: 0.95rem;">python train_script.py</code>
            </div>
            <p style="color: #8b949e; font-size: 0.82rem; margin-top: 1rem;">
                ⏱ ~5–10 min on GTX 1650 · ~350MB VRAM · 2000 iterations
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Architecture Explainer ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 How does NanoGPT work?", expanded=False):
            st.markdown("""
            **NanoGPT** is a character-level **Decoder-Only Transformer** — the same fundamental
            architecture powering GPT-2, GPT-3, and modern LLMs.

            | Component | Role |
            |-----------|------|
            | **Token Embedding** | Maps each character to a dense vector |
            | **Positional Embedding** | Encodes the position of each token |
            | **Multi-Head Attention** | Each token attends to all previous tokens |
            | **Causal Mask** | Prevents attending to future tokens |
            | **Feed-Forward Network** | Applies nonlinear transformation per position |
            | **Residual Connections** | Solves vanishing gradient problem |
            | **Layer Normalization** | Stabilizes training across deep stacks |
            | **LM Head** | Projects hidden states to vocabulary logits |

            During **generation**, the model predicts one character at a time and feeds it back
            as input — this is called **autoregressive decoding**.
            """)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
