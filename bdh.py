import torch
from torch import nn
import torch.nn.functional as F

# HYPERPARAMETERS (Based on Dragon Book Appendix E) [Source 6208]
# N = Number of Neurons. Paper uses 32768. 
# If you have a small GPU/Laptop, reduce N to 4096 or 8192 to avoid OOM.
CONFIG = {
    'D': 256,          # Synaptic dimension (Low-rank)
    'N': 8192,         # Number of Neurons (Particles) - Scalable dimension
    'L': 6,            # Number of Layers
    'H': 4,            # Number of Heads
    'DROPOUT': 0.05,
    'VOCAB_SIZE': 256  # Byte-level tokenization (handles all text)
}

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rope(x, cos, sin):
    return (x * cos) + (torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1) * sin)

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rope = RotaryEmbedding(dim)

    def forward(self, Q, K, V):
        # Apply RoPE to Q and K (Rotary Positional Embeddings) [Source 6211]
        # This handles the "Oscillatory" dynamics of the model
        cos, sin = self.rope(Q)
        Qr = apply_rope(Q, cos, sin)
        Kr = apply_rope(K, cos, sin)
        
        # Linear Attention: (Q @ K.T) @ V with causal masking
        # This implements the "Hebbian Learning" update rule
        energy = torch.matmul(Qr, Kr.transpose(-2, -1))
        
        # Causal mask (cannot see future)
        mask = torch.tril(torch.ones(energy.shape[-2], energy.shape[-1], device=Q.device))
        energy = energy.masked_fill(mask == 0, 0)
        
        return torch.matmul(energy, V)

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        D, N, H = CONFIG['D'], CONFIG['N'], CONFIG['H']
        
        # Parameter-free LayerNorm (Crucial for BDH stability) [Source 6208]
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.wte = nn.Embedding(CONFIG['VOCAB_SIZE'], D)
        self.drop = nn.Dropout(CONFIG['DROPOUT'])
        
        # Trainable Parameters (The "Connectome")
        # E: Encoder, D: Decoders. These form the graph structure.
        self.encoder = nn.Parameter(torch.randn(N, D) * 0.02)
        self.decoder_x = nn.Parameter(torch.randn(H, D, N // H) * 0.02)
        self.decoder_y = nn.Parameter(torch.randn(H, D, N // H) * 0.02)
        self.readout = nn.Parameter(torch.randn(D, CONFIG['VOCAB_SIZE']) * 0.02)
        
        self.attn = LinearAttention(D)

    def forward(self, idx, targets=None, state=None):
        B, T = idx.size()
        
        # Embed input tokens
        v_ast = self.ln(self.wte(idx)) 
        
        # Initialize or load state (for long-context reasoning)
        if state is None:
            # If no state provided, start fresh
            pass
        
        # Universal Layer Loop [Source 6210]
        # The model recycles the same weights L times
        for _ in range(CONFIG['L']):
            # 1. Expansion Phase (D -> N)
            # Projects into high-dimensional neuron space
            # ReLU enforces "Positive Energy" and Sparsity [Source 6002]
            x_raw = torch.einsum('btd,hdm->bthm', v_ast, self.decoder_x)
            x = F.relu(x_raw) 
            
            # 2. Linear Attention (Hebbian Memory)
            # Uses current activation x as both Query and Key
            a_ast = self.attn(Q=v_ast, K=v_ast, V=v_ast) # Simplified Q=K=V for code clarity
            
            # 3. Compression Phase (N -> D)
            # Updates state vector v_ast
            y_raw = torch.einsum('btd,hdm->bthm', self.ln(a_ast), self.decoder_y)
            y = F.relu(y_raw) * x # Gating mechanism
            
            # Project back to low-rank dimension D
            y_flat = y.reshape(B, T, -1)
            update = y_flat @ self.encoder
            v_ast = v_ast + self.ln(update)
            v_ast = self.ln(v_ast)

        # Output predictions
        logits = v_ast @ self.readout
        
        loss = None
        if targets is not None:
            # Calculate Cross Entropy Loss ("Energy")
            loss = F.cross_entropy(logits.view(-1, CONFIG['VOCAB_SIZE']), targets.view(-1))
            
        return logits, loss, v_ast
