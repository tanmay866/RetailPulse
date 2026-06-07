"""Unit tests for src/lstm_lightning.py — Day 6 Lightning LSTM pipeline."""
import numpy as np
import pandas as pd
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.lstm_lightning import (
    RetailDataset,
    RetailDataModule,
    LSTMLightning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_series():
    """100-day synthetic revenue series with a sine trend."""
    np.random.seed(42)
    t = np.arange(200, dtype=np.float32)
    values = 1000 + 200 * np.sin(2 * np.pi * t / 7) + np.random.randn(200) * 50
    return pd.Series(values, index=pd.date_range('2020-01-01', periods=200, freq='D'))


# ---------------------------------------------------------------------------
# Test 1: RetailDataset produces correct tensor shapes
# ---------------------------------------------------------------------------

def test_retail_dataset_shape():
    series = np.arange(50, dtype=np.float32)
    seq_len = 10
    ds = RetailDataset(series, seq_len=seq_len)

    expected_n = len(series) - seq_len
    assert len(ds) == expected_n, f'expected {expected_n} samples, got {len(ds)}'

    x, y = ds[0]
    assert x.shape == (seq_len, 1), f'X shape should be ({seq_len}, 1), got {x.shape}'
    assert y.shape == (1,),         f'y shape should be (1,), got {y.shape}'


# ---------------------------------------------------------------------------
# Test 2: RetailDataModule setup — loaders are non-empty, scaler is fitted
# ---------------------------------------------------------------------------

def test_datamodule_setup(synthetic_series):
    dm = RetailDataModule(synthetic_series, seq_len=10, batch_size=8, val_split=0.1)
    dm.setup()

    assert len(dm.train_dataloader()) > 0, 'train_dataloader must have at least one batch'
    assert len(dm.val_dataloader())   > 0, 'val_dataloader must have at least one batch'

    # scaler must be fitted — transform should not raise
    sample = synthetic_series.values[:5].reshape(-1, 1)
    transformed = dm.scaler.transform(sample)
    assert transformed.shape == (5, 1), 'scaler transform produced unexpected shape'
    assert np.all(np.isfinite(transformed)), 'scaler output contains non-finite values'


# ---------------------------------------------------------------------------
# Test 3: LSTMLightning forward pass — output shape (batch, 1)
# ---------------------------------------------------------------------------

def test_forward_pass():
    model = LSTMLightning(input_size=1, hidden_size=32, num_layers=2, dropout=0.2, lr=1e-3)
    batch_size = 4
    seq_len    = 10
    x = torch.randn(batch_size, seq_len, 1)
    out = model(x)

    assert out.shape == (batch_size, 1), (
        f'expected output shape ({batch_size}, 1), got {out.shape}'
    )
    assert torch.all(torch.isfinite(out)), 'model output contains NaN or Inf'


# ---------------------------------------------------------------------------
# Test 4: Trainer smoke-test — 2 epochs on synthetic data, val_loss logged
# ---------------------------------------------------------------------------

def test_trainer_smoke(synthetic_series):
    dm = RetailDataModule(synthetic_series, seq_len=10, batch_size=8, val_split=0.1)
    model = LSTMLightning(hidden_size=16, num_layers=1, dropout=0.0, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='cpu',
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    assert 'val_loss' in trainer.callback_metrics, (
        'val_loss must appear in trainer.callback_metrics after training'
    )
    val_loss = trainer.callback_metrics['val_loss'].item()
    assert np.isfinite(val_loss), f'val_loss is not finite: {val_loss}'
    assert val_loss > 0,          f'val_loss should be positive, got {val_loss}'
