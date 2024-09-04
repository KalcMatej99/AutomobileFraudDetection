"""Heterogeneous Graph Auto-Encoder."""

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
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GENConv, HeteroConv, TransformerConv
from torch_geometric.typing import Metadata
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

Batch = Any

convs = {
    "gat": GATv2Conv,
    "tr": TransformerConv,
    "gen": GENConv,
}


def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HGATBlock(torch.nn.Module):
    """Heterogeneous Graph Auto-Encoder Encoder."""

    def __init__(  # noqa: PLR0913
        self: "HGATBlock",
        metadata: Metadata,
        edge_sizes: dict[str, int],
        default_hidden_channels_gcn: list[int],
        custom_hidden_channels_gcn: dict[str, list[int]] | None = None,
        aggr_method: str = "mean",
        output_size: dict[str, int] | None = None,
        conv_type_name: str = "gat",
        dropout: float = 0.0,
    ) -> None:
        """Initialize.

        Args:
            metadata (Metadata): Graph metadata
            edge_sizes (dict[str, int]): Edge sizes
            default_hidden_channels_gcn (list[int]): Default hidden channels gcn
            custom_hidden_channels_gcn (dict[str, list[int]] | None, optional): Custom hidden channels gcn. Defaults to None.
            aggr_method (str, optional): Aggregation method. Defaults to "mean".
            output_size (dict[str, int] | None, optional): Output size. Defaults to None.
            conv_type_name (str, optional): Conv type name. Defaults to "gat".
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        _set_seed()
        super().__init__()
        conv_type = convs[conv_type_name]

        additional_kwargs = {}
        if conv_type_name == "gat":
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
            if layer_index < num_layers - 1 or output_size is not None:
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
        if output_size is not None:
            conv = HeteroConv(
                {
                    edge_type: GATv2Conv(
                        (-1, -1),
                        output_size[edge_type[2]],
                        add_self_loops=False,
                        edge_dim=edge_sizes[edge_type[1]],
                    )
                    for edge_type in edges_list
                },
                aggr=aggr_method,
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self: "HGATBlock",
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward.

        Args:
            x_dict (dict[str, torch.Tensor]): X dict
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Edge index dict
            edge_attr_dict (dict[tuple[str, str, str], torch.Tensor]): Edge attr dict

        Returns:
            torch.Tensor: Output tensor
        """
        if len(self.convs) > 1:
            for conv, bns in zip(self.convs[:-1], self.bns, strict=True):
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

                if self.training:
                    x_dict = {key: self.dropout(bns[key](x.relu())) for key, x in x_dict.items()}
                else:
                    x_dict = {key: bns[key](x.relu()) for key, x in x_dict.items()}
        return self.convs[-1](x_dict, edge_index_dict, edge_attr_dict)


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        sigmoid: bool = True,
    ) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for the given node-pairs :obj:`edge_index`.

        Args:
            z1 (torch.Tensor): The latent space :math:`\mathbf{Z_1}`.
            z2 (torch.Tensor): The latent space :math:`\mathbf{Z_2}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z1[edge_index[0]] * z2[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class HGAD(nn.Module):
    """Heterogeneous Graph Aunomaly Detection."""

    def __init__(  # noqa: PLR0913
        self: "HGAD",
        input_sizes: dict[str, int],
        metadata: Metadata,
        edge_sizes: dict[str, int],
        default_hidden_channels_gcn: list[int],
        n_epochs: int,
        step_lr: dict,
        custom_hidden_channels_gcn: dict[str, list[int]] | None = None,
        lr: float = 0.01,
        aggr_method: str = "mean",
        l2_reg: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize.

        Args:
            input_sizes (dict[str, int]): Input sizes
            metadata (Metadata): Graph metadata
            edge_sizes (dict[str, int]): Edge sizes
            default_hidden_channels_gcn (list[int]): Default hidden channels gcn
            n_epochs (int): Number of epochs
            step_lr (dict): Step lr
            custom_hidden_channels_gcn (dict[str, list[int]] | None, optional): Custom hidden channels gcn. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 0.01.
            aggr_method (str, optional): Aggregation method. Defaults to "mean".
            l2_reg (float, optional): L2 regularization. Defaults to 0.1.
            dropout (float, optional): Dropout. Defaults to 0.0.
        """
        super().__init__()
        self.encoder = HGATBlock(
            metadata=metadata,
            edge_sizes=edge_sizes,
            default_hidden_channels_gcn=default_hidden_channels_gcn,
            custom_hidden_channels_gcn=custom_hidden_channels_gcn,
            aggr_method=aggr_method,
            dropout=dropout,
        )
        self.feature_decoder = HGATBlock(
            metadata=metadata,
            edge_sizes=edge_sizes,
            default_hidden_channels_gcn=default_hidden_channels_gcn[::-1][1:],
            custom_hidden_channels_gcn={
                node_type: custom_hidden_channels_gcn[node_type][::-1][1:] for node_type in custom_hidden_channels_gcn
            }
            if custom_hidden_channels_gcn
            else None,
            aggr_method=aggr_method,
            output_size=input_sizes,
            dropout=dropout,
        )

        self.link_decoder = InnerProductDecoder()

        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.n_epochs = n_epochs
        self.scheduler = None
        if step_lr["use"]:
            self.scheduler = StepLR(self.optimizer, step_size=step_lr["step_size"], gamma=step_lr["gamma"])

    def forward(
        self: "HGAD",
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x_dict (dict[str, torch.Tensor]): X dict
            edge_index_dict (dict[tuple[str, str, str], torch.Tensor]): Edge index dict
            edge_attr_dict (dict[tuple[str, str, str], torch.Tensor]): Edge attr dict

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensors
        """
        x = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return x, self.feature_decoder(x, edge_index_dict, edge_attr_dict)

    def fit(  # noqa: PLR0915
        self: "HGAD",
        data_loader: DataLoader,
        val_data_loader: DataLoader | None = None,
        early_stopping: dict | None = None,
        split_name: str | None = None,
    ) -> None:
        """Fit.

        Args:
            data_loader (DataLoader): data loader
            val_data_loader (DataLoader, optional): validation data loader. Defaults to None.
            early_stopping (dict | None, optional): Early stopping. Defaults to None.
            split_name (str | None, optional): Split name. Defaults to None.
        """
        criterion = nn.MSELoss()
        epoch_bar = tqdm(range(1, self.n_epochs + 1), desc="Fitting HGAD...")
        best_val_auc = 0
        best_model_params = self.state_dict()
        for epoch in epoch_bar:
            self.train()

            total_examples = total_loss = 0
            predictions: list[np.ndarray] = []
            trues: list[np.ndarray] = []
            for batch in data_loader:
                batch_size = batch["claim"].batch_size
                loss, batch_predictions, batch_trues = self._batch_fit(batch, criterion)

                total_examples += batch_size
                total_loss += loss * batch_size

                predictions.append(batch_predictions)
                trues.append(batch_trues)
            if self.scheduler:
                self.scheduler.step()
            epoch_loss = total_loss / total_examples
            mlflow.log_metric(f"{split_name}/train_loss", epoch_loss, step=epoch)

            epoch_true_labels = np.concatenate(trues)
            epoch_predictions_proba = np.concatenate(predictions)
            epoch_predictions_labels = epoch_predictions_proba > 1 / 2
            auc_score = roc_auc_score(epoch_true_labels, epoch_predictions_proba) * 100
            accuracy = accuracy_score(epoch_true_labels, epoch_predictions_labels) * 100
            mlflow.log_metric(f"{split_name}/train_auc", auc_score, step=epoch)
            mlflow.log_metric(f"{split_name}/train_ca", accuracy, step=epoch)

            if epoch == 1:
                total_params = sum(p.numel() for p in self.parameters())
                logging.info(f"Total number of parameters: {total_params}")

            if val_data_loader:
                val_predictions, val_trues, val_feature_loss, val_link_loss = self.predict(val_data_loader, return_all=True)
                val_auc_score = roc_auc_score(val_trues, val_predictions) * 100
                val_accuracy = accuracy_score(val_trues, val_predictions > 1 / 2) * 100
                mlflow.log_metric(f"{split_name}/val_feature_loss", val_feature_loss, step=epoch)
                mlflow.log_metric(f"{split_name}/val_link_loss", val_link_loss, step=epoch)
                mlflow.log_metric(f"{split_name}/val_auc", val_auc_score, step=epoch)
                mlflow.log_metric(f"{split_name}/val_ca", val_accuracy, step=epoch)
                epoch_bar.set_postfix(
                    loss=epoch_loss,
                    auc=auc_score,
                    ca=accuracy,
                    val_loss=val_feature_loss,
                    val_auc=val_auc_score,
                    val_ca=val_accuracy,
                )

                if early_stopping:
                    if val_auc_score > best_val_auc:
                        best_val_auc = val_auc_score
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

    def _link_bce_loss(self: "HGAD", z: torch.Tensor, pos_edge_indexs: dict) -> torch.Tensor:
        eps = 1e-15
        losses = None
        valid_edges = 0
        for edge_type in pos_edge_indexs:
            pos_edge_index = pos_edge_indexs[edge_type]
            if pos_edge_index.size(1) == 0:
                continue

            edge_from = edge_type[0]
            edge_to = edge_type[2]
            z1 = z[edge_from]
            z2 = z[edge_to]
            valid_edges += 1
            pos_loss = -torch.log(self.link_decoder(z1, z2, pos_edge_index, sigmoid=True) + eps).mean()

            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=(z1.size(0), z2.size(0)))
            neg_loss = -torch.log(1 - self.link_decoder(z1, z2, neg_edge_index, sigmoid=True) + eps).mean()

            loss = pos_loss + neg_loss
            if losses is None:
                losses = loss
            else:
                losses += loss
        if losses is None:
            raise ValueError("No valid edges")
        return losses / valid_edges

    def _batch_fit(self: "HGAD", batch: Batch, feature_criterion: nn.MSELoss) -> tuple[float, np.ndarray, np.ndarray]:
        """Batch fit.

        Args:
            batch (Batch): Batch
            feature_criterion (nn.MSELoss): Criterion

        Returns:
            tuple[float, np.ndarray, np.ndarray]: Loss, predictions, true labels
        """
        self.train()
        self.optimizer.zero_grad()
        batch_size = batch["claim"].batch_size
        enc_out, feature_out = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        loss = feature_criterion(feature_out["claim"], batch["claim"].x) + self._link_bce_loss(enc_out, batch.edge_index_dict)
        loss.backward()
        self.optimizer.step()
        errors = ((feature_out["claim"][:batch_size] - batch["claim"].x[:batch_size]) ** 2).mean(dim=1).squeeze().detach().cpu().numpy()
        batch_predictions = errors.reshape(-1, 1)
        batch_predictions /= np.max(batch_predictions)
        batch_trues = batch["claim"].y[:batch_size].squeeze().cpu().numpy()

        return loss.item(), batch_predictions, batch_trues

    def predict(
        self: "HGAD",
        data_loader: DataLoader,
        *,
        return_all: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | np.ndarray:
        """Predict.

        Args:
            data_loader (DataLoader): DataLoader
            return_all (bool, optional): Return all. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, float] | np.ndarray: Predicted values
        """
        self.eval()
        criterion = nn.MSELoss()
        list_errors = []
        trues = []
        total_examples = total_feature_loss = total_link_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch_size = batch["claim"].batch_size
                enc_out, feature_out = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                list_errors.append(
                    ((feature_out["claim"][:batch_size] - batch["claim"].x[:batch_size]) ** 2).mean(dim=1).squeeze().cpu().numpy(),
                )
                if return_all:
                    feature_loss = criterion(feature_out["claim"], batch["claim"].x)
                    link_loss = self._link_bce_loss(enc_out, batch.edge_index_dict)
                    total_examples += batch_size
                    total_feature_loss += feature_loss.item() * batch_size
                    total_link_loss += link_loss.item() * batch_size

                    trues.append(batch["claim"].y[:batch_size].squeeze().cpu().detach().numpy())
            if return_all:
                feature_loss = total_feature_loss / total_examples
                link_loss = total_link_loss / total_examples

        errors = np.concatenate(list_errors)

        errors = errors.reshape(-1, 1)
        errors /= np.max(errors)
        if return_all:
            return errors, np.concatenate(trues), feature_loss, link_loss
        return errors

    def predict_anomaly(self: "HGAD", features: torch.Tensor) -> np.ndarray:
        """Predict anomaly.

        Args:
            features (torch.Tensor): Features

        Returns:
            np.ndarray: Predicted values
        """
        errors = ((self(features) - features) ** 2).mean(dim=1).squeeze().cpu().numpy()
        errors = errors.reshape(-1, 1)
        errors /= np.max(errors)
        return errors
