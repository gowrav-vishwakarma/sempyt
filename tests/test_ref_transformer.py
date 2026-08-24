"""GPT zoo == pands TransformerLM with copied weights."""

import sys
from pathlib import Path

import torch

from sempyt.zoo.gpt import GPTConfig, GPTLM
from sempyt.testing import assert_close

_PANDS = Path("/home/gowrav/Development/pands")
if str(_PANDS) not in sys.path:
    sys.path.insert(0, str(_PANDS))

from v1.transformer_baseline import TransformerConfig, TransformerLM  # noqa: E402


def test_gpt_matches_pands_baseline():
    cfg = TransformerConfig(
        vocab_size=32, max_seq_len=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.0
    )
    ref = TransformerLM(cfg).eval()
    ours = GPTLM(
        GPTConfig(
            vocab_size=32, max_seq_len=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.0
        )
    ).eval()
    missing = ours.load_state_dict(ref.state_dict(), strict=True)
    assert missing.missing_keys == [] or True  # load_state_dict returns IncompatibleKeys
    ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        a = ref(ids)
        b = ours(ids)
    assert_close(a, b, rtol=1e-5, atol=1e-6)
