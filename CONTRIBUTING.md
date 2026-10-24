# Contributing

Contributions are welcome! Please follow these guidelines.

## Development Setup

```bash
git clone https://github.com/Danuka97/tinynn_project.git
cd tinynn-project
pip install -e ".[dev]"
```

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep the library's core principle intact: **all
   backpropagation is implemented manually** — do not introduce calls to
   `torch.autograd` or `Tensor.backward()` in `tinynn/` or `models/`.
3. Add or update tests in `tests/` for any behavior change.
4. Ensure the full test suite passes: `pytest`.
5. Open a pull request with a clear description of the change.

## Code Style

- Match the existing style: plain functions and small classes with
  explanatory comments describing the math being implemented.
- Every layer must implement `forward` and `backward`, and expose
  `parameters()` / `grads()` where it owns trainable tensors.

## Reporting Issues

Open an issue with a minimal reproduction, the command you ran, and the full
error output.
