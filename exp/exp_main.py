"""Standard experiment runner for Lyra forecasting models."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from models.Lyra import Model
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


log = logging.getLogger(__name__)


class Exp_Main:
    """Train, validate, and evaluate a forecasting model."""

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self) -> torch.device:
        use_cuda = bool(getattr(self.args, "use_gpu", False))
        if use_cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{getattr(self.args, 'gpu', 0)}")
        else:
            device = torch.device("cpu")
        log.info("Use device: %s", device)
        return device

    def _build_model(self) -> nn.Module:
        return Model(self.args)

    def _prepare_batch(self, batch):
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        return batch_x, batch_y

    def _prediction_and_target(self, batch):
        batch_x, batch_y = self._prepare_batch(batch)
        outputs = self.model(batch_x)
        horizon = int(self.args.pred_len)
        prediction = outputs[:, -horizon:, :]
        target = batch_y[:, -horizon:, :]
        return prediction, target

    @torch.no_grad()
    def _validate(self, loader, criterion: nn.Module) -> float:
        self.model.eval()
        losses = []
        for batch in loader:
            prediction, target = self._prediction_and_target(batch)
            losses.append(float(criterion(prediction, target).detach().cpu()))
        if not losses:
            raise RuntimeError("Validation loader produced no batches.")
        return float(np.mean(losses))

    def train(self, setting: str):
        """Train the model and restore the best validation checkpoint."""
        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        _, train_loader = data_provider(self.args, "train")
        _, val_loader = data_provider(self.args, "val")

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.0),
        )
        criterion = nn.HuberLoss(delta=1.0)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )

        for epoch in range(int(self.args.train_epochs)):
            self.model.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, target = self._prediction_and_target(batch)
                loss = criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss is not finite.")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if not losses:
                raise RuntimeError("Training loader produced no batches.")

            train_loss = float(np.mean(losses))
            val_loss = self._validate(val_loader, criterion)
            log.info(
                "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f",
                epoch + 1,
                self.args.train_epochs,
                train_loss,
                val_loss,
            )

            early_stopping(val_loss, self.model, str(checkpoint_dir))
            if early_stopping.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        return self.model

    def train_e2e(self, setting: str):
        """Compatibility alias for direct end-to-end training."""
        return self.train(setting)

    @torch.no_grad()
    def test(self, setting: str, load_checkpoint: bool = False):
        """Evaluate the model and save metrics and predictions."""
        _, test_loader = data_provider(self.args, "test")
        if load_checkpoint:
            checkpoint_path = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        self.model.eval()
        predictions = []
        targets = []
        for batch in test_loader:
            prediction, target = self._prediction_and_target(batch)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

        if not predictions:
            raise RuntimeError("Test loader produced no batches.")

        predictions_array = np.concatenate(predictions, axis=0)
        targets_array = np.concatenate(targets, axis=0)
        mae, mse, rmse, mape, mspe = metric(predictions_array, targets_array)

        result_dir = Path("results") / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_dir / "pred.npy", predictions_array)
        np.save(result_dir / "true.npy", targets_array)

        with open("results.csv", "a", encoding="utf-8") as result_file:
            result_file.write(
                f"{setting}, MSE: {mse:.4f}, MAE: {mae:.4f}{os.linesep}"
            )

        log.info("MSE: %.6f | MAE: %.6f", mse, mae)
        return mse, mae
