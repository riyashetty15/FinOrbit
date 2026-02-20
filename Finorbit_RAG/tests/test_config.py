import pytest
from config import EmbeddingConfig


def test_embedding_dimension_override():
    cfg = EmbeddingConfig(model_name='FinanceMTEB/FinE5')
    assert cfg.dimension == 768


def test_default_finance_model():
    cfg = EmbeddingConfig()
    assert 'philschmid' in cfg.model_name or 'bge' in cfg.model_name
