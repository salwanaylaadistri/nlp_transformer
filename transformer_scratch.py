import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def causal_mask(seq_len):
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    additive = (1.0 - mask) * -1e9
    return additive 

# Positional: RoPE class
class RotaryPositionEmbedding:
    def __init__(self, head_dim, max_len=512):
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
        positions = np.arange(max_len)
        freqs = np.outer(positions, inv_freq) 
        self.cos = np.cos(freqs).astype(np.float32)  
        self.sin = np.sin(freqs).astype(np.float32)

    def apply(self, x):
        # x: [batch, heads, seq, head_dim]
        b, h, seq, d = x.shape
        assert d == self.head_dim
        half = d // 2
        cos = self.cos[:seq]  
        sin = self.sin[:seq]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        x = x.reshape(b, h, seq, half, 2)
        x_even = x[..., 0] 
        x_odd = x[..., 1]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        x_rot = np.stack([x_rot_even, x_rot_odd], axis=-1).reshape(b, h, seq, d)
        return x_rot

# Components 
class TokenEmbedding:
    def __init__(self, vocab_size, d_model, rng):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rng = rng
        self.W = rng.normal(scale=0.02, size=(vocab_size, d_model)).astype(np.float32)
    
    def __call__(self, x):
        return self.W[x]  # [batch, seq, d_model]

class PositionalEncodingSinusoidal:
    def __init__(self, d_model, max_len=512):
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe
    
    def __call__(self, x):
        batch, seq, d = x.shape
        return x + self.pe[:seq, :][np.newaxis, :, :]

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.d_model = d_model
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)
    
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class ScaledDotProductAttention:
    def __init__(self, dropout=0.0):
        self.dropout = dropout
    
    def __call__(self, Q, K, V, attn_mask=None):
        # Q, K, V: [batch, heads, seq, head_dim]
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores = scores + attn_mask[None, None, :, :]
        attn_probs = softmax(scores, axis=-1)
        output = np.matmul(attn_probs, V)
        return output, attn_probs

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng, rotary=None):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rng = rng
        self.W_q = rng.normal(scale=0.02, size=(d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(scale=0.02, size=(d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(scale=0.02, size=(d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(scale=0.02, size=(d_model, d_model)).astype(np.float32)
        self.attn = ScaledDotProductAttention()
        self.rotary = rotary  
    
    def split_heads(self, x):
        b, seq, d = x.shape
        x = x.reshape(b, seq, self.n_heads, self.head_dim)
        return x.transpose(0,2,1,3)
    
    def combine_heads(self, x):
        b, heads, seq, head_dim = x.shape
        x = x.transpose(0,2,1,3).reshape(b, seq, heads*head_dim)
        return x
    
    def __call__(self, x, attn_mask=None, return_attn=False):
        # x: [batch, seq, d_model]
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Qh = self.split_heads(Q)  # [b, heads, seq, head_dim]
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)
        
        if self.rotary is not None:
            Qh = self.rotary.apply(Qh)
            Kh = self.rotary.apply(Kh)
        attn_out, attn_probs = self.attn(Qh, Kh, Vh, attn_mask=attn_mask)
        combined = self.combine_heads(attn_out)
        out = combined @ self.W_o
        if return_attn:
            return out, attn_probs
        return out

class FeedForward:
    def __init__(self, d_model, d_ff, rng):
        self.d_model = d_model
        self.d_ff = d_ff
        self.rng = rng
        self.W1 = rng.normal(scale=0.02, size=(d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = rng.normal(scale=0.02, size=(d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros((d_model,), dtype=np.float32)
    
    def __call__(self, x):
        inter = x @ self.W1 + self.b1
        gelu = inter * 0.5 * (1.0 + np.tanh(np.sqrt(2/np.pi) * (inter + 0.044715 * inter**3)))
        out = gelu @ self.W2 + self.b2
        return out

class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff, rng, rotary=None):
        self.mha = MultiHeadAttention(d_model, n_heads, rng, rotary=rotary)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)
        self.ln2 = LayerNorm(d_model)
    
    def __call__(self, x, attn_mask=None, return_attn=False):
        x_norm = self.ln1(x)
        if return_attn:
            mha_out, attn_probs = self.mha(x_norm, attn_mask=attn_mask, return_attn=True)
        else:
            mha_out = self.mha(x_norm, attn_mask=attn_mask, return_attn=False)
        x = x + mha_out
        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        if return_attn:
            return x, attn_probs
        return x

class DecoderOnlyTransformer:
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=8,
        d_ff=512,
        n_layers=2,
        max_len=128,
        rng=None,
        positional="sinusoidal",  
        weight_tying=False
    ):
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_len = max_len
        self.positional = positional
        self.weight_tying = weight_tying

        # Embedding
        self.tok_emb = TokenEmbedding(vocab_size, d_model, rng)

        # Positional encodings
        if self.positional == "sinusoidal":
            self.pos_enc = PositionalEncodingSinusoidal(d_model, max_len=max_len)
            rotary = None
        elif self.positional == "rope":
            head_dim = d_model // n_heads
            self.pos_enc = None
            rotary = RotaryPositionEmbedding(head_dim, max_len=max_len)
        else:
            raise ValueError("positional must be 'sinusoidal' or 'rope'")

        # Blocks 
        self.blocks = [DecoderBlock(d_model, n_heads, d_ff, rng, rotary=rotary) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)

        # Output projection
        if weight_tying:
            self.W_out = self.tok_emb.W.T
        else:
            self.W_out = rng.normal(scale=0.02, size=(d_model, vocab_size)).astype(np.float32)

    def forward(self, tokens, return_attn=False, return_all_attn=False):
        batch, seq = tokens.shape
        assert seq <= self.max_len
        x = self.tok_emb(tokens)  # [batch, seq, d_model]
        if self.positional == "sinusoidal":
            x = self.pos_enc(x)

        mask = causal_mask(seq)

        attn_weights_all = [] if return_all_attn else None
        attn_last = None

        for block in self.blocks:
            if return_attn or return_all_attn:
                x, attn = block(x, attn_mask=mask, return_attn=True)
                attn_last = attn
                if return_all_attn:
                    attn_weights_all.append(attn)
            else:
                x = block(x, attn_mask=mask, return_attn=False)

        x = self.ln_f(x)
        logits = x @ self.W_out  # [batch, seq, vocab]
        probs_last = softmax(logits[:, -1, :], axis=-1)
        if return_all_attn:
            return logits, probs_last, attn_weights_all
        if return_attn:
            return logits, probs_last, attn_last
        return logits, probs_last

# Visualization helper 
def visualize_attention(attn_probs, batch_idx=0, head_idx=0, show=True, save_path=None):
    """
    attn_probs: [batch, heads, seq, seq]
    Displays a single attention heatmap for (batch_idx, head_idx).
    """
    a = attn_probs[batch_idx, head_idx]
    plt.figure()
    plt.imshow(a)
    plt.colorbar()
    plt.title(f"Attention map (batch {batch_idx}, head {head_idx})")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# Demo 
def run_demo(positional="sinusoidal", weight_tying=False, visualize=True):
    rng = np.random.default_rng(42)
    np.random.seed(42)
    vocab_size = 50
    batch = 2
    seq_len = 6
    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 2

    tokens = rng.integers(low=0, high=vocab_size, size=(batch, seq_len))
    print("Input tokens:\n", tokens)

    model = DecoderOnlyTransformer(
        vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_len=128,
        rng=rng,
        positional=positional,
        weight_tying=weight_tying
    )

    logits, probs_last, attn = model.forward(tokens, return_attn=True)

    print("\nLogits shape:", logits.shape)
    print("Probs last shape:", probs_last.shape)
    sums = probs_last.sum(axis=-1)
    print("Softmax sum (per batch) at last position:", sums)

    print("\nAttention probs shape (from last block):", attn.shape)
    a = attn[0, 0]
    print("\nAttention probs (batch 0, head 0):\n", np.round(a, 4))
    upper_triangle = np.triu(a, k=1)
    max_future_attn = upper_triangle.max()
    print("Max attention weight to future positions (should be near 0):", max_future_attn)

    assert logits.shape == (batch, seq_len, vocab_size)
    assert probs_last.shape == (batch, vocab_size)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-5)
    assert max_future_attn < 1e-5 or np.isclose(max_future_attn, 0.0, atol=1e-4)

    print("\nAll checks passed!")

    if visualize:
        visualize_attention(attn, batch_idx=0, head_idx=0, show=True)

if __name__ == "__main__":
    # Sinusoidal positional encoding, no weight tying
    run_demo(positional="sinusoidal", weight_tying=False, visualize=True)

    # RoPE with weight tying
    # run_demo(positional="rope", weight_tying=True, visualize=True)
