"""Tests for the byte-level BPE tokenizer."""
from tokenizer import BPETokenizer


def test_train_learns_merges():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 20, vocab_size=280)
    assert len(tok.merges) == 280 - 256


def test_train_stops_early_on_short_corpus():
    tok = BPETokenizer()
    tok.train("ab", vocab_size=300)  # not enough pairs for all merges
    assert len(tok.merges) <= 1


def test_encode_produces_valid_ids():
    tok = BPETokenizer()
    vocab_size = 300
    tok.train("hello world, hello there, hello!" * 10, vocab_size=vocab_size)

    ids = tok.encode("hello world")
    assert all(0 <= i < vocab_size for i in ids)


def test_encode_compresses_repeated_text():
    tok = BPETokenizer()
    tok.train("abcabcabcabc" * 10, vocab_size=270)

    raw_len = len("abcabcabcabc".encode("utf-8"))
    encoded = tok.encode("abcabcabcabc")
    assert 0 < len(encoded) < raw_len


def test_encode_unseen_text_falls_back_to_bytes():
    tok = BPETokenizer()
    tok.train("aaaa bbbb " * 10, vocab_size=260)

    ids = tok.encode("xyz")  # no learned merges apply
    assert ids == list("xyz".encode("utf-8"))
