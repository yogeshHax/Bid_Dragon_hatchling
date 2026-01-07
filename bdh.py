import torch
from torch import nn
import torch.nn.functional as F
import math

# CONFIGURATION (The "Scale-Free" Parameters)
# If you have a good GPU (A100/V100/3090): Use N = 32768 or 65536
# If you have a laptop CPU: Use N = 2048 or 4096
CONFIG = {
    'D': 256,         # Synaptic dimension (Low-rank)
    'N': 8192,        # Number of Neurons (Particles)
    'L': 6,           # Number of Layers (Universal Transformer depth)
    'H': 4,           # Heads
    'DROPOUT': 0.05,
    'VOCAB_SIZE': 256 # Byte-level tokenization (Handles all text)
}

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: [batch, seq_len, head_dim]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rope = RotaryEmbedding(dim)

    def forward(self, Q, K, V):
        # Apply RoPE (Oscillator Dynamics - Source 103)
        cos, sin = self.rope(Q)
        Qr = apply_rope(Q, cos, sin)
        Kr = apply_rope(K, cos, sin)
        
        # Linear Attention: (Q @ K.T) @ V
        # The .tril() enforces causality (can't see future)
        # We use standard matrix multiplication for parallel GPU training
        energy = torch.matmul(Qr, Kr.transpose(-2, -1))
        mask = torch.tril(torch.ones_like(energy))
        energy = energy.masked_fill(mask == 0, 0)
        
        return torch.matmul(energy, V)

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        D, N, H = CONFIG['D'], CONFIG['N'], CONFIG['H']
        
        # Parameter-free LayerNorm (Source 81)
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.wte = nn.Embedding(CONFIG['VOCAB_SIZE'], D)
        self.drop = nn.Dropout(CONFIG['DROPOUT'])
        
        # The Graph Parameters (Matrices E, Dx, Dy)
        # Initialized with normal distribution to simulate random graph
        self.encoder = nn.Parameter(torch.randn(N, D) * 0.02)
        self.decoder_x = nn.Parameter(torch.randn(H, D, N // H) * 0.02)
        self.decoder_y = nn.Parameter(torch.randn(H, D, N // H) * 0.02)
        self.readout = nn.Parameter(torch.randn(D, CONFIG['VOCAB_SIZE']) * 0.02)
        
        self.attn = LinearAttention(D)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # 1. Embed Input
        v_ast = self.ln(self.wte(idx)) 
        
        # 2. Universal Layer Loop (Source 69)
        # Reusing the same "Brain" weights L times
        for _ in range(CONFIG['L']):
            # Expansion Phase (Positive Energy Rule - Source 41)
            # We reshape to handle heads
            x_raw = torch.einsum('btd,hdm->bthm', v_ast, self.decoder_x)
            x = F.relu(x_raw) # Sparsity enforcement
            
            # Linear Attention (Hebbian Learning Simulation)
            # Q=x, K=x (Self-resonance)
            # We flatten heads for attention then reshape back
            x_flat = x.reshape(B, T, -1)
            # Note: In a full impl, we'd project x_flat to D size for attention
            # Here we simplify to run direct dynamics on the latent D
            
            # Attention Mechanism
            a_ast = self.attn(v_ast, v_ast, v_ast) # Simplified for stability
            
            # Compression Phase (Update State)
            # y = ReLU(LN(a_ast) @ Dy) * x
            y_raw = torch.einsum('btd,hdm->bthm', self.ln(a_ast), self.decoder_y)
            y = F.relu(y_raw) * x # Gating
            
            # Back to State Vector v_ast
            y_flat = y.reshape(B, T, -1)
            update = y_flat @ self.encoder
            v_ast = v_ast + self.ln(update)

        # 3. Prediction (Readout)
        logits = v_ast @ self.readout
        
        loss = None
        if targets is not None:
            # Calculate "Surprisal" (Energy)
            loss = F.cross_entropy(logits.view(-1, CONFIG['VOCAB_SIZE']), targets.view(-1))
            
        return logits, loss
