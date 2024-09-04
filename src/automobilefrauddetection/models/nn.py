"""Models."""

import copy
import logging
import os
import random
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import Linear
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
load_dotenv()
device = torch.device(f"cuda:{os.getenv('CUDA_INDEX')}" if torch.cuda.is_available() else "cpu")
logging.info("Device: %s", device)

Batch = Any


def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleNN(torch.nn.Module):
    """NN model."""

    def __init__(  # noqa: PLR0913
        self: "SimpleNN",
        hidden_channels_linear: list[int],
        n_epochs: int,
        step_lr: dict,
        lr: float = 0.01,
        dropout: float = 0.0,
        l2_reg: float = 0.1,
    ) -> None:
        """Initialize.

        Args:
            metadata (Metadata): Graph metadata
            edge_sizes (dict[str, int]): Edge sizes
            default_hidden_channels_gcn (list[int]): Default hidden channels gcn
            hidden_channels_linear (list[int]): Hidden channels linear
            n_epochs (int): Number of epochs
            step_lr (dict): Step lr
            custom_hidden_channels_gcn (dict[str, list[int]] | None, optional): Custom hidden channels gcn. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 0.01.
            aggr_method (str, optional): Aggregation method. Defaults to "mean".
            dropout (float, optional): Dropout percentage. Defaults to 0.0.
            l2_reg (float, optional): L2 regularization. Defaults to 0.1.
        """
        _set_seed()
        super().__init__()

        self.linears = nn.ModuleList(
            [Linear(-1, hidden_channel_linear) for hidden_channel_linear in hidden_channels_linear] + [Linear(-1, 1)],
        )
        self.lin_bns = nn.ModuleList([nn.BatchNorm1d(hidden_channel_linear) for hidden_channel_linear in hidden_channels_linear])

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.n_epochs = n_epochs
        self.scheduler = None
        if step_lr["use"]:
            self.scheduler = StepLR(self.optimizer, step_size=step_lr["step_size"], gamma=step_lr["gamma"])

    def forward(
        self: "SimpleNN",
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for lin, bns in zip(self.linears[:-1], self.lin_bns, strict=True):
            x = bns(lin(x).relu())
            x = self.dropout(x) if self.training else x

        return self.sigmoid(self.linears[-1](x))

    def fit(  # noqa: PLR0915, PLR0912, C901
        self: "SimpleNN",
        data_loader: DataLoader,
        val_data_loader: DataLoader | None = None,
        early_stopping: dict | None = None,
        split_name: str | None = None,
    ) -> None:
        """Fit.

        Args:
            data_loader (DataLoader): Data loader
            val_data_loader (DataLoader | None, optional): Validation data loader. Defaults to None.
            early_stopping (dict, optional): Early stopping configuration. Defaults to None.
            split_name (str, optional): Split name. Defaults to None.
        """
        criterion = nn.BCELoss()
        epoch_bar = tqdm(range(1, self.n_epochs + 1), desc="Fitting Simple NN...")
        best_val_loss = np.inf
        best_model_params = self.state_dict()
        for epoch in epoch_bar:
            self.train()

            total_examples = total_loss = 0
            predictions: list[np.ndarray] = []
            trues: list[np.ndarray] = []
            for batch in data_loader:
                batch_size = batch[0].shape[0]
                loss, batch_predictions, batch_trues = self._batch_fit(batch, criterion)

                total_examples += batch_size
                total_loss += loss * batch_size

                predictions.append(batch_predictions)
                trues.append(batch_trues)
            if self.scheduler:
                self.scheduler.step()
            epoch_loss = total_loss / total_examples
            if split_name:
                mlflow.log_metric(f"{split_name}/train_loss", epoch_loss, step=epoch)

            epoch_true_labels = np.concatenate(trues)
            epoch_predictions_proba = np.concatenate(predictions)
            epoch_predictions_labels = epoch_predictions_proba > 1 / 2
            auc_score = roc_auc_score(epoch_true_labels, epoch_predictions_proba) * 100
            accuracy = accuracy_score(epoch_true_labels, epoch_predictions_labels) * 100
            if split_name:
                mlflow.log_metric(f"{split_name}/train_auc", auc_score, step=epoch)
                mlflow.log_metric(f"{split_name}/train_ca", accuracy, step=epoch)

            if epoch == 1:
                total_params = sum(p.numel() for p in self.parameters())
                logging.info(f"Total number of parameters: {total_params}")

            if val_data_loader:
                val_predictions, val_trues, val_loss = self.predict_data_loader(val_data_loader, return_all=True)
                val_auc_score = roc_auc_score(val_trues, val_predictions) * 100
                val_accuracy = accuracy_score(val_trues, val_predictions > 1 / 2) * 100
                if split_name:
                    mlflow.log_metric(f"{split_name}/val_loss", val_loss, step=epoch)
                    mlflow.log_metric(f"{split_name}/val_auc", val_auc_score, step=epoch)
                    mlflow.log_metric(f"{split_name}/val_ca", val_accuracy, step=epoch)
                epoch_bar.set_postfix(
                    loss=epoch_loss,
                    auc=auc_score,
                    ca=accuracy,
                    val_loss=val_loss,
                    val_auc=val_auc_score,
                    val_ca=val_accuracy,
                )

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        n_iter_with_no_improvment = 0
                        best_model_params = copy.deepcopy(self.state_dict())

                    else:
                        n_iter_with_no_improvment += 1
                        if epoch >= early_stopping["not_before_epoch"] and n_iter_with_no_improvment >= early_stopping["patience"]:
                            logging.info(f"Early stopping at iteration {epoch}")
                            self.load_state_dict(best_model_params)
                            break

            else:
                epoch_bar.set_postfix(loss=epoch_loss, auc=auc_score, ca=accuracy)

    def _batch_fit(self: "SimpleNN", batch: Batch, criterion: nn.BCELoss) -> tuple[float, np.ndarray, np.ndarray]:
        """Batch fit.

        Args:
            batch (Batch): Batch
            criterion (nn.BCELoss): Criterion

        Returns:
            tuple[float, np.ndarray, np.ndarray]: Loss, predictions, true labels
        """
        self.train()
        self.optimizer.zero_grad()
        batch_features, batch_target = batch
        out = self(batch_features)
        loss = criterion(out.squeeze(), batch_target.squeeze())
        loss.backward()
        self.optimizer.step()

        batch_predictions = out.squeeze().cpu().detach().numpy()
        batch_trues = batch_target.squeeze().cpu().numpy()

        return loss.item(), batch_predictions, batch_trues

    def predict(
        self: "SimpleNN",
        features: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        *,
        return_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float] | np.ndarray:
        """Predict.

        Args:
            features (pd.DataFrame | np.ndarray): Features
            y (np.ndarray): True labels
            return_all (bool): If true return also true labels and loss. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, float] | np.ndarray: Predicted values
        """
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        test_features_tensor = torch.from_numpy(features).to(torch.float32).to(device)
        if y is None:
            y = np.zeros(features.shape[0])
        test_target_tensor = torch.from_numpy(y).to(torch.float32).to(device)
        test_data_loader = DataLoader(
            TensorDataset(test_features_tensor, test_target_tensor),
            batch_size=test_features_tensor.shape[0],
            shuffle=False,
        )
        return self.predict_data_loader(test_data_loader, return_all=return_all)

    def predict_data_loader(
        self: "SimpleNN",
        data_loader: DataLoader,
        *,
        return_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float] | np.ndarray:
        """Predict.

        Args:
            data_loader (DataLoader): data loader
            return_all (bool): If true return also true labels and loss. Defaults to False.

        Returns:
            np.ndarray: Predicted values
        """
        self.eval()
        criterion = nn.BCELoss()
        predictions = []
        trues = []
        total_examples = total_loss = 0
        with torch.no_grad():
            for batch_features, batch_target in data_loader:
                batch_size = batch_features.shape[0]
                out = self(batch_features)
                loss = criterion(out.squeeze(), batch_target.squeeze())
                total_examples += batch_size
                total_loss += loss.item() * batch_size

                predictions.append(out.squeeze().cpu().detach().numpy())
                trues.append(batch_target.squeeze().cpu().numpy())
            loss = total_loss / total_examples

        if return_all:
            return np.concatenate(predictions), np.concatenate(trues), loss
        return np.concatenate(predictions)
