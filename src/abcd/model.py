from torch import nn, Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import MLP
from lightning import LightningModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from torchmetrics.functional import (
    r2_score,
    auroc,
    average_precision,
)
from abcd.config import Config, Targets, Tasks


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


class Network(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: SGD,
        scheduler: ReduceLROnPlateau,
        target: Targets,
        task: Tasks,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target: Targets = target
        self.task: Tasks = task
        # self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, batch: tuple[Tensor, Tensor]):
        inputs, labels = batch
        outputs = self(inputs)
        mask = ~labels.isnan()
        labels = labels[mask]  # TODO .unsqueeze(1)
        outputs = outputs[mask]
        return outputs, labels

    def regression_metrics(self, outputs, labels) -> dict:
        loss = self.criterion(outputs, labels)
        r2 = r2_score(outputs, labels, multioutput="raw_values").nanmean()
        return {"loss": loss, "r2": r2}

    def classification_metrics(self, outputs, labels) -> dict:
        loss = self.criterion(outputs, labels)
        auroc_score = auroc(outputs, labels, task=self.target)
        ap = average_precision(outputs, labels, task=self.target)
        return {"loss": loss, "auroc": auroc_score, "ap": ap}

    def training_step(self, batch, batch_idx):
        outputs, labels = self.step(batch)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, labels = self.step(batch)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs, labels = self.step(batch)
        if self.task == "regression":
            metrics = self.regression_metrics(outputs, labels)
        elif self.task == "classification":
            metrics = self.classification_metrics(outputs, labels)
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        return outputs, labels

    def configure_optimizers(self):
        scheduler_config = {
            "scheduler": self.scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 2,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_config}


def make_trainer(config: Config) -> tuple[Trainer, ModelCheckpoint]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.filepaths.checkpoints,
        filename="{epoch}_{step}_{val_loss:.2f}",
        save_top_k=1,
        verbose=config.verbose,
        monitor="val_loss",
        mode="min",
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
    )
    logger = TensorBoardLogger(save_dir=config.filepaths.logs)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    callbacks = [
        checkpoint_callback,
        early_stopping,
        RichProgressBar(),
    ]
    return (
        Trainer(
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=config.training.gradient_clip,
            max_epochs=config.training.max_epochs,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=config.logging.log_every_n_steps,
            fast_dev_run=config.fast_dev_run,
            check_val_every_n_epoch=1,
        ),
        checkpoint_callback,
    )


def make_criterion(task: str):
    match task:
        case "classification":
            return nn.CrossEntropyLoss()
        case "regression":
            return nn.MSELoss()
        case _:
            raise ValueError(
                f"Invalid task '{task}'. Choose from: 'classification' or 'regression'"
            )


def make_model(
    method: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
):
    match method:
        case "rnn":
            return RNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        case "mlp":
            hidden_channels = [hidden_dim] * num_layers
            return MLP(
                in_channels=input_dim,
                hidden_channels=hidden_channels + [output_dim],
                dropout=dropout,
            )
        case _:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'rnn' or 'mlp'")


def make_network(
    method: str,
    task: Tasks,
    target: Targets,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
):
    model = make_model(
        method=method,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    criterion = make_criterion(task=task)
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)
    return Network(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        task=task,
        target=target,
    )
