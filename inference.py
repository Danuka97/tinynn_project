# inference.py
import torch
from models.gpt import GPT
from tokenizer import BPETokenizer

def load_checkpoint(path, device="cpu"):
    """Rebuilds the model and tokenizer that train.py saved."""
    checkpoint = torch.load(path, map_location=device)

    model = GPT(**checkpoint["model_config"], device=device)
    for param, saved in zip(model.parameters(), checkpoint["params"], strict=True):
        param.copy_(saved)

    tokenizer = BPETokenizer(merges=checkpoint["merges"])
    return model, tokenizer

def generate_text(model, token_ids, max_new_tokens, stop_token_id=None, temperature=1.0):
    token_ids = list(token_ids)  # Work on a copy so the caller's prompt is left alone
    device = model.lm_head.weight.device

    for _ in range(max_new_tokens):
        # 1. Forward pass on the most recent tokens that fit in the model's
        # context window, as a batch of one: [1, seq_len]
        context = torch.tensor(token_ids[-model.max_seq_len:], device=device).unsqueeze(0)
        logits = model(context)

        # 2. Get the last position's logits and apply temperature
        last_word_logits = logits[0, -1]
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

# Example Setup (after running train.py):
# model, tokenizer = load_checkpoint("gpt_model.pt")
# output = generate_text(model, tokenizer.encode("The meaning of life is"), max_new_tokens=50, temperature=0.8)
# print(tokenizer.decode(output))
