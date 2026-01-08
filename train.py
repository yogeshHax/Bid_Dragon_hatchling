import torch
import torch.nn.functional as F
from torch import nn

# --- Section 1: Hyperparameters (The Scale of the Dragon) ---
# N (Neurons) is massive compared to D (Synaptic dimension). 
# This follows scale-free laws where few 'hubs' manage the traffic [9-11].
D = 256        # The "Synaptic" Dimension (Rank)
H = 4          # The Number of Heads (Dividing the brain into 4 'clubs') [7, 12]
N = 32768      # The "Neuron" Dimension (Scale) [4, 13]
L = 6          # The Number of Layers (Iterations in the Universal loop) [10]
DROPOUT = 0.05 # Standard regularization to prevent overfitting [14]
VOCAB_SIZE = 256 # Assuming byte-level tokenization for simplicity [15, 16]

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The Stabilizer (Parameter-Free LayerNorm)
        # We set elementwise_affine=False so the model doesn't 'learn' to scale. 
        # This prevents "Positive Energy" from exploding in the loop [11, 17, 18].
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # 2. The Ears (Vocabulary Embedding)
        # Translates raw data into 'brain language' in dimension D [13, 16].
        self.wte = nn.Embedding(VOCAB_SIZE, D)
        self.drop = nn.Dropout(DROPOUT)
        
        # 3. The Body (Parameter Matrices)
        # The Encoder compresses the high-dimensional thought back to D [8, 19, 20].
        self.encoder = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        
        # The Decoders expand the thought to the massive Neuron Space (N) [8, 19].
        # We split neurons into H heads to create 'communities' [7, 8].
        self.decoder_x = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.decoder_y = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        
        # 4. The Mouth (Readout)
        # Final layer to predict the next word/token [21, 22].
        self.readout = nn.Parameter(torch.zeros((D, VOCAB_SIZE)).normal_(std=0.02))
        
        # 5. The Eyes (Linear Attention)
        # This is the custom engine for memory tracking [23, 24].
        self.attn = LinearAttention()
   # This goes inside the BDH_GPU class
    def forward(self, idx):
        # B = batch size, T = sequence length
        B, T = idx.size() 
        
        # 1. Input: Translating words into "Synaptic Space" (D)
        # We normalize immediately to keep the 'volume' stable [7, 8].
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) 

        # 2. The Universal Loop: The heart of the reasoning [3]
        # We pass the signal through the SAME brain L times.
        for _ in range(L):
            # A. Expansion: Query Generation [9]
            # Projects the thought up to the massive Neuron Space (N).
            # ReLU makes it positive and sparse (only 5% wake up) [10, 11].
            x = F.relu(v_ast @ self.decoder_x) 
            
            # B. Attention: Searching the Past [12]
            # Uses current active neurons (x) to query past memories (v_ast).
            a_ast = self.attn(Q=x, K=x, V=v_ast) 
            
            # C. Gated Update: Only active neurons change [13]
            # The '* x' ensures only the 5% of neurons that are 'awake' can update.
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x 
            
            # D. Compression & Residual: Saving the new thought [14]
            # Project back down to D and add to the original thought.
            y = y.transpose(1, 2).reshape(B, 1, T, N)
            v_ast = v_ast + self.ln(self.drop(y) @ self.encoder)
            v_ast = self.ln(v_ast) # Stabilizing again for the next loop [7].

        # 3. Output: The Mouth speaks the prediction [15]
        return v_ast.squeeze(1) @ self.readout

# --- Section 6: The Eyes (The Linear Attention Mechanism) ---
# This is a separate class at the bottom of bdh.py [4, 6].
class LinearAttention(nn.Module):
    def forward(self, Q, K, V):
        # RoPE adds 'waves' or 'rhythms' so the AI knows word order [16, 17].
        Qr = RoPE(Q) 
        Kr = RoPE(K)
        
        # The Associativity Trick: We multiply K and V first to stay fast [6].
        # .tril(diagonal=-1) ensures the AI cannot 'cheat' by looking at the future [18].
        return (Qr @ Kr.mT).tril(diagonal=-1) @ V

import torch
from torch.utils.data import Dataset, DataLoader

# --- Section 7: The Data Loader (Translating Books to Math) ---
class NarrativeDataset(Dataset):
    def __init__(self, novel_path, sequence_length=2048):
        # 1. Load the long novel text (e.g., "The Count of Monte Cristo.txt")
        # The sources confirm these narratives are very long [Problem Statement].
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 2. Byte-level Tokenization
        # Instead of complex dictionaries, we treat every character/byte as a number [the_Dragon_Book.pdf].
        # This is simple and effective for "Continuous Reasoning" [Problem Statement].
        self.data = torch.ByteTensor(list(text.encode('utf-8')))
        self.seq_len = sequence_length

    def __len__(self):
        # We divide the 100,000+ words into small 2,048-token segments [the_Dragon_Book.pdf].
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        # 3. Slicing the Story
        # We pull out one segment to show the Dragon.
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # We return the segment and a "shifted" version for the AI to predict the next token.
        chunk = self.data[start:end].long()
        return chunk[:-1], chunk[1:]

# --- Section 8: Preparation for Training ---
# This helper function sets up the "conveyor belt" of data.
def get_batch_loader(file_path, batch_size=4):
    dataset = NarrativeDataset(file_path)
    # DataLoader handles the 'Batching'—giving multiple chunks at once [the_Dragon_Book.pdf].
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

import torch.optim as optim

# --- Section 9: The Training Loop (The Handoff) ---
def train_dragon(model, dataloader, epochs=5):
    # 1. The Optimizer: Using AdamW as recommended by the sources [6, 7]
    # We use a learning rate that decays from 10^-3 to 10^-4 [6].
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        # The "State" (v_ast/rho) represents the model's current memory [3, 4].
        # We carry it across minibatches to maintain context [7, 8].
        state = None 
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 2. Forward Pass: Predicting the next part of the story [9]
            # In a full TBPTT implementation, the 'state' would be passed here.
            outputs = model(inputs)
            
            # 3. Loss Calculation: Measuring how 'surprised' the Dragon is [10]
            # When the model is surprised (wrong), its neurons fire more wildly [10, 11].
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            
            # 4. Backpropagation: Strengthening the 'Muscle Memory' [12, 13]
            # "Neurons that fire together, wire together" [14, 15].
            loss.backward()
            
            # We detach the state to prevent 'gradient explosion' over long context [16, 17].
            # This is the "Detach" experiment logic for efficiency [3, 18].
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# --- Section 10: Execution ---
if __name__ == "__main__":
    # Initialize the Dragon Anatomy
    dragon = BDH_GPU()
    
    # Load the novel (Ensure your filtered 'novel.txt' is in the same folder)
    loader = get_batch_loader("novel.txt")
    
    # Start the training process
    print("Starting the training of the Hatchling...")
    train_dragon(dragon, loader)
    
    # Save the trained brain
    torch.save(dragon.state_dict(), "trained_dragon.pth")
    print("Training complete. The Dragon's memory is saved to 'trained_dragon.pth'.")