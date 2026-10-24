"""Tests for text generation and checkpoint loading."""
import torch

from inference import generate_text, load_checkpoint
from models.gpt import GPT
from tokenizer import BPETokenizer
from train import save_checkpoint

MODEL_CONFIG = dict(vocab_size=32, max_seq_len=8, embed_size=16, num_layers=2)


def make_model():
    return GPT(**MODEL_CONFIG, device="cpu")


def test_generate_from_a_list_prompt():
    torch.manual_seed(0)
    prompt = [1, 2, 3]

    output = generate_text(make_model(), prompt, max_new_tokens=5)

    assert prompt == [1, 2, 3]  # the caller's list is left alone
    assert output[:3] == prompt
    assert len(output) == 8
    assert all(0 <= t < 32 for t in output)


def test_generate_past_the_context_window():
    torch.manual_seed(0)
    # 21 tokens is far more than max_seq_len=8, so the model must only see the latest 8
    output = generate_text(make_model(), [1], max_new_tokens=20)
    assert len(output) == 21


def test_generate_stops_at_stop_token():
    model = make_model()
    model.lm_head.bias[7] = 100.0  # makes token 7 (practically) certain

    output = generate_text(model, [1, 2], max_new_tokens=10, stop_token_id=7)
    assert output == [1, 2, 7]


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = make_model()
    tokenizer = BPETokenizer()
    tokenizer.train("hello world, hello there " * 10, vocab_size=270)

    path = tmp_path / "gpt_model.pt"
    save_checkpoint(path, model, MODEL_CONFIG, tokenizer)
    loaded_model, loaded_tokenizer = load_checkpoint(path)

    for p, loaded in zip(model.parameters(), loaded_model.parameters(), strict=True):
        assert torch.equal(p, loaded)
    assert loaded_tokenizer.encode("hello there") == tokenizer.encode("hello there")
