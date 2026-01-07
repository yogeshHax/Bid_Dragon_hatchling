import torch
import pandas as pd
from bdh import BDH_GPU, CONFIG

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_energy(model, text):
    """
    Calculates the 'Energy' (Loss) of a text sequence.
    High Energy = Surprising/Contradictory.
    Low Energy = Consistent.
    """
    # Tokenize
    ids = [ord(c) for c in text if ord(c) < 256]
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()

def solve():
    # 1. Load the Pre-trained Brain
    model = BDH_GPU().to(DEVICE)
    try:
        model.load_state_dict(torch.load("bdh_brain.pt"))
        print("Brain loaded successfully.")
    except FileNotFoundError:
        print("Please run train_ingest.py first!")
        return
    model.eval()

    # 2. Load the Challenge (Simulating reading train.pdf)
    # In real usage, use a PDF reader library or copy the text
    challenges = [
        # From Train.pdf (True case)
        {"id": 93, "text": "Noirtier secretly raised the Southern Army..."},
        # From Train.pdf (False case - Contradiction)
        {"id": 105, "text": "Noirtier was a loyal servant of the King and hated Napoleon."} 
    ]
    
    print("-" * 50)
    print("DETECTING CONTRADICTIONS (Thermodynamic Method)")
    print("-" * 50)

    # 3. Energy Thresholding
    # Calculate energy for each claim
    results = []
    for c in challenges:
        energy = get_energy(model, c['text'])
        results.append((c['id'], c['text'], energy))
        print(f"Claim ID {c['id']}: Energy = {energy:.4f}")

    # Determine Threshold (Simple mean for demonstration)
    # In competition, train a logistic regression on these energy scores
    avg_energy = sum(r[2] for r in results) / len(results)
    print(f"\nAverage System Energy: {avg_energy:.4f}")
    
    print("\nPREDICTIONS:")
    for pid, text, energy in results:
        # If energy is HIGHER than average, it's likely a lie (High Surprise)
        # Prediction 0 = Contradiction, 1 = Consistent
        if energy > avg_energy:
            pred = 0 
            status = "CONTRADICTION"
        else:
            pred = 1
            status = "CONSISTENT"
            
        print(f"[{status}] ID {pid}: {pred}")

if __name__ == "__main__":
    solve()