# inference.py
import torch
from models.gpt import GPT 

def generate_text(model, token_ids, max_new_tokens, stop_token_id, temperature=1.0):
    for _ in range(max_new_tokens):
        # 1. Forward pass
        logits = model(token_ids) 
        
        # 2. Get last word and apply temperature
        last_word_logits = logits[-1]
        last_word_logits = last_word_logits / temperature
        
        # 3. Convert to probabilities
        probs = torch.softmax(last_word_logits, dim=-1)
        
        # 4. Roll the weighted die!
        predicted_token = torch.multinomial(probs, num_samples=1).item()
        
        # 5. Add to sequence
        token_ids.append(predicted_token)
        
        # 6. Stop early if finished
        if predicted_token == stop_token_id:
            break
            
    return token_ids

# Example Setup:
# model = GPT(vocab_size=50000, max_seq_len=1024, embed_size=256, num_layers=4, device="cpu")
# output = generate_text(model, [45, 912], max_new_tokens=50, stop_token_id=50256, temperature=0.8)
# print(output)