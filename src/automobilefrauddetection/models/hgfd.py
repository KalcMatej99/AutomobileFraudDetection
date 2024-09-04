"""Models."""

import copy
import logging
import random
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv, GATConv, HeteroConv, Linear, TransformerConv
from torch_geometric.typing import Metadata
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

Batch = Any

convs = {
    "gat": GATConv,
    "gatv2": GATv2Conv,
    "tr": TransformerConv,
}


def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HeteroGNN(torch.nn.Module):
    """GNN model."""

    def __init__(  # noqa: PLR0913
        self: "HeteroGNN",
        metadata: Metadata,
        edge_sizes: dict[str, int],
        default_hidden_channels_gcn: list[int],
        hidden_channels_linear: list[int],
        n_epochs: int,
        step_lr: dict,
        custom_hidden_channels_gcn: dict[str, list[int]] | None = None,
        lr: float = 0.01,
        aggr_method: str = "mean",
        dropout: float = 0.0,
        l2_reg: float = 0.1,
        conv_type_name: str = "gat",
        criterion: str = "bce",
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
            conv_type_name (str, optional): Conv type name. Defaults to "gat".
            criterion (str, optional): Criterion. Defaults to "bce".
        """
        _set_seed()
        super().__init__()

        conv_type = convs[conv_type_name]

        additional_kwargs = {}
        if conv_type_name in ("gat", "gatv2"):
            additional_kwargs = {"add_self_loops": False}

        edges_list: list = metadata[1]
        num_layers: int = len(default_hidden_channels_gcn)
        if custom_hidden_channels_gcn is None:
            custom_hidden_channels_gcn = {}

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer_index in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: conv_type(
                        (-1, -1),
                        custom_hidden_channels_gcn[edge_type[2]][layer_index],
                        edge_dim=edge_sizes[edge_type[1]],
                        **additional_kwargs,
                    )
                    if edge_type[2] in custom_hidden_channels_gcn
                    else conv_type(
                        (-1, -1),
                        default_hidden_channels_gcn[layer_index],
                        edge_dim=edge_sizes[edge_type[1]],
                        **additional_kwargs,
                    )
                    for edge_type in edges_list
                },
                aggr=aggr_method,
            )
            self.convs.append(conv)
            self.bns.append(
                nn.ModuleDict(
                    {
                        node_type: nn.BatchNorm1d(custom_hidden_channels_gcn[node_type][layer_index])
                        if node_type in custom_hidden_channels_gcn
                        else nn.BatchNorm1d(default_hidden_channels_gcn[layer_index])
                        for node_type in metadata[0]
                    },
                ),
            )

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
        if criterion == "bce":
            self.criterion = nn.BCELoss()
        elif criterion == "bbce":
            self.criterion = self.balanced_bce_loss
        else:
            raise ValueError(f"Criterion {criterion} not supported.")

    def forward(
        self: "HeteroGNN",
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Forward.

        Args:
            x_dict (dict[str, torch.Tensor]): Node features
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Edge indices
            edge_attr_dict (dict[tuple[str, str, str], torch.Tensor]): Edge attributes

        Returns:
            torch.Tensor: Output tensor
        """
        for conv, bns in zip(self.convs, self.bns, strict=True):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            if self.training:
                x_dict = {key: self.dropout(bns[key](x.relu())) for key, x in x_dict.items()}
            else:
                x_dict = {key: bns[key](x.relu()) for key, x in x_dict.items()}

        x = x_dict["claim"]

        for lin, bns in zip(self.linears[:-1], self.lin_bns, strict=True):
            x = bns(lin(x).relu())
            x = self.dropout(x) if self.training else x

        return self.sigmoid(self.linears[-1](x))

    def balanced_bce_loss(
        self: "HeteroGNN",
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: float | None = None,
        neg_weight: float | None = None,
    ) -> torch.Tensor:
        """Modified Binary Cross-Entropy Loss for handling class imbalance.

        Args:
            predictions (torch.Tensor): The predicted logits (raw output scores) of shape (N,).
            targets (torch.Tensor): The ground truth binary labels of shape (N,).
            pos_weight (float, optional): The weight for the positive class. If None, it defaults to 1.
            neg_weight (float, optional): The weight for the negative class. If None, it defaults to 1.

        Returns:
            torch.Tensor: The calculated modified BCE loss.
        """
        # If pos_weight and neg_weight are not provided, set them to 1 (standard BCE Loss)
        if pos_weight is None:
            pos_weight = 1.0
        if neg_weight is None:
            neg_weight = 1.0

        # Calculate sigmoid activation on predictions (logits)
        probs = torch.sigmoid(predictions)

        # Calculate the BCE loss for positive and negative examples separately
        pos_loss = -pos_weight * targets * torch.log(probs + 1e-8)  # Adding epsilon to avoid log(0)
        neg_loss = -neg_weight * (1 - targets) * torch.log(1 - probs + 1e-8)  # Adding epsilon to avoid log(0)

        # Combine both positive and negative losses
        loss = pos_loss + neg_loss

        # Average the loss over the batch
        return loss.mean()

    def fit(  # noqa: PLR0915, PLR0912, C901
        self: "HeteroGNN",
        data_loader: DataLoader,
        val_data_loader: DataLoader | None = None,
        early_stopping: dict | None = None,
        split_name: str | None = None,
    ) -> None:
        """Fit.

        Args:
            data_loader (DataLoader): data loader
            val_data_loader (DataLoader, optional): validation data loader. Defaults to None.
            early_stopping (dict, optional): Early stopping configuration. Defaults to None.
            split_name (str, optional): Split name. Defaults to None.
        """
        epoch_bar = tqdm(range(1, self.n_epochs + 1), desc="Fitting Hetero GNN...")
        best_val_loss = np.inf
        best_model_params = self.state_dict()
        for epoch in epoch_bar:
            self.train()

            total_examples = total_loss = 0
            predictions: list[np.ndarray] = []
            trues: list[np.ndarray] = []
            for batch in data_loader:
                batch_size = batch["claim"].batch_size
                loss, batch_predictions, batch_trues = self._batch_fit(batch)

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
                val_predictions, val_trues, val_loss = self.predict(val_data_loader, return_all=True)
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

    def _batch_fit(self: "HeteroGNN", batch: Batch) -> tuple[float, np.ndarray, np.ndarray]:
        """Batch fit.

        Args:
            batch (Batch): Batch
            criterion (nn.BCELoss): Criterion

        Returns:
            tuple[float, np.ndarray, np.ndarray]: Loss, predictions, true labels
        """
        self.train()
        self.optimizer.zero_grad()
        batch_size = batch["claim"].batch_size
        out = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        if isinstance(self.criterion, nn.BCELoss):
            loss = self.criterion(out[:batch_size].squeeze(), batch["claim"].y[:batch_size].squeeze())
        else:
            pos_weight = float(batch["claim"].y[:batch_size].squeeze().mean().cpu())
            loss = self.criterion(out[:batch_size].squeeze(), batch["claim"].y[:batch_size].squeeze(), 1 - pos_weight, pos_weight)
        loss.backward()
        self.optimizer.step()

        batch_predictions = out[:batch_size].squeeze().cpu().detach().numpy()
        batch_trues = batch["claim"].y[:batch_size].squeeze().cpu().numpy()

        return loss.item(), batch_predictions, batch_trues

    def predict(
        self: "HeteroGNN",
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
        predictions = []
        trues = []
        total_examples = total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch_size = batch["claim"].batch_size
                out = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                predictions.append(out[:batch_size].squeeze().cpu().detach().numpy())
                if return_all:
                    if isinstance(self.criterion, nn.BCELoss):
                        loss = self.criterion(out[:batch_size].squeeze(), batch["claim"].y[:batch_size].squeeze())
                    else:
                        pos_weight = float(batch["claim"].y[:batch_size].squeeze().mean().cpu())
                        loss = self.criterion(
                            out[:batch_size].squeeze(),
                            batch["claim"].y[:batch_size].squeeze(),
                            1 - pos_weight,
                            pos_weight,
                        )
                    total_examples += batch_size
                    total_loss += loss.item() * batch_size

                    trues.append(batch["claim"].y[:batch_size].squeeze().cpu().numpy())
            if return_all:
                loss = total_loss / total_examples

        if return_all:
            return np.concatenate(predictions), np.concatenate(trues), loss
        return np.concatenate(predictions)
