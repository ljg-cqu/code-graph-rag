from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from codebase_rag.unixcoder import UniXcoder


def test_forward_uses_2d_attention_mask() -> None:
    model = UniXcoder.__new__(UniXcoder)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(pad_token_id=1)

    token_embeddings = torch.randn(2, 4, 3)
    transformer = MagicMock(return_value=(token_embeddings,))
    model.model = transformer

    source_ids = torch.tensor([[0, 2, 3, 1], [0, 4, 1, 1]])
    returned_tokens, sentence_embeddings = UniXcoder.forward(model, source_ids)

    assert tuple(returned_tokens.shape) == (2, 4, 3)
    assert tuple(sentence_embeddings.shape) == (2, 3)

    attention_mask = transformer.call_args.kwargs["attention_mask"]
    assert tuple(attention_mask.shape) == (2, 4)
