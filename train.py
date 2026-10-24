# train.py
import argparse
import yaml
import torch
import logging
from datasets import load_dataset


# Import your custom modules
from models.gpt import GPT

from tokenizer import BPETokenizer

def setup_logging():
    """Configures logging to both the terminal and a text file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler("training_log.txt"),
            logging.StreamHandler()
        ]
    )

def get_batch(data, batch_size, block_size, device):
    """Slices a random batch of inputs and targets from the dataset."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def main(config):
    setup_logging()
    logging.info(f"Starting GPT training pipeline using {config['dataset']}...")

    # 1. Hardware Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using hardware device: {device}")

    # 2. Extract Hyperparameters
    vocab_size = config["vocab_size"]
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    epochs = config["epochs"]

    # 3. Initialize Model and Tokenizer
    model = GPT(
        vocab_size=vocab_size, 
        max_seq_len=block_size, 
        embed_size=256, 
        num_layers=4, 
        device=device
    )
    tokenizer = BPETokenizer()

    # 4. Data Pipeline
    logging.info(f"Loading {config['dataset']} dataset (1% slice)...")
    raw_dataset = load_dataset(config['dataset'], split="train[:1%]")

    # Train tokenizer on a sample
    sample_text = "".join(raw_dataset[:50]["text"])
    logging.info("Training Tokenizer...")
    tokenizer.train(sample_text, vocab_size=vocab_size)

    # Tokenize the entire dataset
    def tokenize_function(example):
        return {"input_ids": tokenizer.encode(example["text"])}

    logging.info("Tokenizing dataset (this may take a moment)...")
    tokenized_dataset = raw_dataset.map(tokenize_function, remove_columns=["text"])

    # Flatten into one continuous tensor
    all_tokens = []
    for item in tokenized_dataset:
        all_tokens.extend(item["input_ids"])
    data_tensor = torch.tensor(all_tokens, dtype=torch.long)
    logging.info(f"Total tokens ready for training: {len(data_tensor)}")

    # 5. The Training Loop (manual backward pass + manual SGD update,
    # since this framework's layers implement their own backward() methods
    # rather than relying on PyTorch autograd)
    logging.info("Beginning optimization loop...")
    for step in range(epochs):
        X, Y = get_batch(data_tensor, batch_size, block_size, device)

        # Forward pass
        logits = model(X)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = Y.view(-1)
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)

        # Manual backward pass: gradient of softmax cross-entropy w.r.t. logits
        # is (softmax(logits) - one_hot(targets)) / N
        probs = torch.softmax(flat_logits, dim=-1)
        d_flat_logits = probs.clone()
        d_flat_logits[torch.arange(flat_targets.shape[0]), flat_targets] -= 1.0
        d_flat_logits /= flat_targets.shape[0]
        dLogits = d_flat_logits.view(logits.shape)

        model.backward(dLogits)

        # Manual SGD update using each layer's manually computed *_grad buffers
        with torch.no_grad():
            for p, g in zip(model.parameters(), model.grads()):
                p -= lr * g

        if step % 100 == 0:
            logging.info(f"Step {step:04d} | Loss: {loss.item():.4f}")

    logging.info("Training complete!")
    
    # 6. Save the model weights
    torch.save(model.parameters(), "gpt_model.pt")
    logging.info("Model weights successfully saved to gpt_model.pt!")

if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Train a custom GPT model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file."
    )
    
    # Parse the terminal commands
    args = parser.parse_args()
    
    # Load the specific YAML file requested
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)
        
    # Run the pipeline
    main(config_dict)