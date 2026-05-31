"""PyTorch Lightning LSTM for daily revenue forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.preprocessing import MinMaxScaler


class RetailDataset(Dataset):
    """Sliding-window dataset over a scaled 1-D series."""

    def __init__(self, series: np.ndarray, seq_len: int = 30):
        values = np.array(series, dtype=np.float32)
        X, y = [], []
        for i in range(len(values) - seq_len):
            X.append(values[i:i + seq_len])
            y.append(values[i + seq_len])
        self.X = torch.tensor(np.array(X)).unsqueeze(-1)   # (N, seq_len, 1)
        self.y = torch.tensor(np.array(y)).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RetailDataModule(pl.LightningDataModule):
    """Scales, sequences, and splits a revenue series into train/val loaders."""

    def __init__(self, series: pd.Series, seq_len: int = 30,
                 batch_size: int = 32, val_split: float = 0.1):
        super().__init__()
        self.series = series
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        values = self.series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values).flatten()

        split = int(len(scaled) * (1 - self.val_split))
        train_scaled = scaled[:split]
        val_scaled = scaled[split:]

        self._train_ds = RetailDataset(train_scaled, self.seq_len)
        self._val_ds = RetailDataset(val_scaled, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self._train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0)


class LSTMLightning(pl.LightningModule):
    """Stacked LSTM defined directly inside LightningModule."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }


def build_trainer(max_epochs: int = 50, patience: int = 10,
                  run_name: str = 'lstm-lightning',
                  checkpoint_dir: str = 'models') -> pl.Trainer:
    mlflow_logger = MLFlowLogger(
        experiment_name='retailpulse-forecasting',
        run_name=run_name,
    )
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='lstm_lightning_checkpoint',
                monitor='val_loss',
                save_top_k=1,
                mode='min',
            ),
        ],
        logger=mlflow_logger,
        enable_progress_bar=True,
    )


def train_lstm_lightning(
    train_series: pd.Series,
    seq_len: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 10,
) -> tuple[LSTMLightning, RetailDataModule]:
    """Scale, sequence, and train LSTM with Lightning. Returns (model, datamodule)."""
    dm = RetailDataModule(train_series, seq_len=seq_len,
                          batch_size=batch_size, val_split=0.1)
    model = LSTMLightning(hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, lr=lr)
    trainer = build_trainer(max_epochs=max_epochs, patience=patience)
    trainer.fit(model, datamodule=dm)
    return model, dm


def forecast_lstm_lightning(
    model: LSTMLightning,
    seed_series: pd.Series,
    steps: int,
    scaler: MinMaxScaler,
    seq_len: int = 30,
) -> np.ndarray:
    """Autoregressive multi-step forecast. Returns predictions in original scale."""
    model.eval()
    scaled = scaler.transform(seed_series.values.reshape(-1, 1)).flatten()
    window = list(scaled[-seq_len:])
    preds_scaled = []

    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(x).item()
            preds_scaled.append(pred)
            window.append(pred)

    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
