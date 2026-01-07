import torch
import os
from bdh import BDH_GPU, CONFIG

# --- SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE} (Particles: {CONFIG['N']})")

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def train_model():
    # 1. Load the "Big Dataset"
    # Ensure 'The Count of Monte Cristo.txt' is in the folder
    filename = "The Count of Monte Cristo.txt" 
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please download the novel text.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Simple Byte Tokenization (Robustness - Source 100)
    data = torch.tensor([ord(c) for c in text if ord(c) < 256], dtype=torch.long)
    print(f"Loaded {len(data)} tokens.")

    # 3. Initialize Dragon
    model = BDH_GPU().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # 4. Training Loop (Hebbian Rewiring)
    model.train()
    BATCH_SIZE = 8
    BLOCK_SIZE = 128
    STEPS = 500 # Increase this for better results (e.g., 5000)

    print("Beginning Hebbian Rewiring (Training)...")
    for i in range(STEPS):
        xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

    # 5. Save the "Brain State"
    torch.save(model.state_dict(), "bdh_brain.pt")
    print("Model saved as 'bdh_brain.pt'")

if __name__ == "__main__":
    train_model()