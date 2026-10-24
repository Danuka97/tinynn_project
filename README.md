# TinyNN: A From-Scratch Neural Network Library

[![Tests](https://github.com/Danuka97/tinynn_project/actions/workflows/ci.yml/badge.svg)](https://github.com/Danuka97/tinynn_project/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)](pyproject.toml)

TinyNN is an educational deep-learning library that implements a GPT-style
language model **without relying on automatic differentiation**. Every layer
implements its own analytic `forward` and `backward` pass on top of raw
PyTorch tensors — PyTorch is used purely as a tensor library, never for
autograd. The result is a small, readable codebase that makes the mechanics of
backpropagation through a Transformer fully explicit.

## Features

- **Manual backpropagation** — every module (`LinearLayer`, `SelfAttention`,
  `TransformerBlock`, `EmbeddingLayer`, …) derives and implements its own
  gradients.
- **GPT language model** — token + positional embeddings, stacked causal
  self-attention transformer blocks, and an LM head ([models/gpt.py](models/gpt.py)).
- **Byte-level BPE tokenizer** — trained from scratch on raw UTF-8 bytes
  ([tokenizer.py](tokenizer.py)).
- **Optimizers** — SGD and Adam implemented against the library's manual
  gradient buffers ([tinynn/optim](tinynn/optim)).
- **End-to-end training pipeline** — config-driven training on OpenWebText via
  [train.py](train.py), and temperature-sampling text generation via
  [inference.py](inference.py).

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/Danuka97/tinynn_project.git
cd tinynn_project
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Quickstart

### Training

Hyperparameters live in [config.yaml](config.yaml):

```yaml
dataset: "Skylion007/openwebtext"
vocab_size: 5000
block_size: 128
batch_size: 16
learning_rate: 0.0003  # sized for Adam, which train.py uses
steps: 1000            # optimizer steps (one random batch each), not passes over the data
```

Then run:

```bash
python train.py --config config.yaml
```

The script trains the BPE tokenizer on a sample of the dataset, tokenizes the
corpus, and trains the model with manual backpropagation and Adam. Progress is
logged to the console and `training_log.txt`. The trained weights, the model
config and the tokenizer's merges are saved together to `gpt_model.pt`.

### Generation

```python
from inference import generate_text, load_checkpoint

# Rebuilds the model and tokenizer saved by train.py
model, tokenizer = load_checkpoint("gpt_model.pt")

prompt = tokenizer.encode("The meaning of life is")
output = generate_text(model, prompt, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output))
```

## How It Works

Instead of `loss.backward()`, TinyNN threads gradients explicitly:

1. The cross-entropy gradient w.r.t. the logits is computed analytically as
   `softmax(logits) - one_hot(y)`, scaled by `1/N`.
2. `model.backward(dLogits)` propagates the gradient backwards through the LM
   head, transformer blocks, and embedding tables, with each layer caching its
   inputs during `forward` and consuming them in `backward`.
3. Optimizers update parameters using the manually populated `*_grad`
   buffers — no autograd graph is ever built.

## Repository Layout

```
tinynn/            Core library
  nn/              Layers: linear, attention, transformer, embeddings, losses
  optim/           Optimizers: SGD, Adam
models/gpt.py      GPT model assembled from tinynn modules
tokenizer.py       Byte-level BPE tokenizer
train.py           Config-driven training entry point
inference.py       Temperature-sampling text generation
test_run.py        Minimal end-to-end smoke example (linear regression)
tests/             Pytest suite
```

## Running the Tests

```bash
pytest
```

The suite includes numerical gradient checks that validate every layer's manual
`backward`, up to the full GPT, against PyTorch autograd.

## Citing

If you use this code in your work, please cite:

```bibtex
@software{tinynn_project,
  title  = {TinyNN: A From-Scratch Neural Network Library},
  author = {Danuka Wickramasinghe},
  year   = {2026},
  url    = {https://github.com/Danuka97/tinynn_project}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
