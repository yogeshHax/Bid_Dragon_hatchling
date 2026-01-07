import torch
import os
import glob
from bdh import BDH_GPU, CONFIG

# Detect hardware
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}")

def train():
    # 1. Load Data (The Big Dataset)
    # Reads all .txt files in current directory
    text_data = ""
    files = glob.glob("*.txt")
    if not files:
        print("Error: No .txt files found! Please place the novels in this folder.")
        return

    for file in files:
        print(f"Loading {file}...")
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            text_data += f.read() + "\n"
    
    print(f"Total training tokens: {len(text_data)}")

    # 2. Tokenize (Bytes)
    # Simple robust tokenization for raw text
    data = torch.tensor([ord(c) for c in text_data if ord(c) < 256], dtype=torch.long)
    
    # 3. Initialize Model
    model = BDH_GPU().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # 4. Training Loop (Truncated BPTT)
    # We train in chunks, but carry state forward to simulate long-term reading
    model.train()
    BLOCK_SIZE = 128
    BATCH_SIZE = 4
    STEPS = 2000 # Increase for better accuracy (e.g. 5000+)
    
    print("Wiring the brain (Training)...")
    for step in range(STEPS):
        # Get random batch
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(DEVICE)
        
        # Forward pass
        logits, loss, _ = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss/Energy = {loss.item():.4f}")
            
    # 5. Save the "Brain"
    torch.save(model.state_dict(), "bdh_brain.pth")
    print("Training complete. Brain saved as 'bdh_brain.pth'.")

if __name__ == "__main__":
    train()
