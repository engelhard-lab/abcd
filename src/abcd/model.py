from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from torchmetrics import R2Score
from abcd.config import Config


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
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


# def make_model(method: str):
#     match method:
#         case "rnn":
#             return RNN(
#                 input_dim=input_dim,
#                 output_dim=output_dim,
#                 hidden_dim=hidden_dim,
#                 num_layers=num_layers,
#                 dropout=dropout,
#             )
#         case "mlp":
#             return MLPModel(
#                 input_dim=input_dim,
#                 hidden_dim=hidden_dim,
#                 num_layers=num_layers,
#                 dropout=dropout,
#                 output_dim=output_dim,
#             )
#         case "linear":
#             return LinearModel(input_dim=input_dim, output_dim=output_dim)


class Network(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = RNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.criterion = nn.MSELoss()
        self.train_metric = R2Score(num_outputs=output_dim, multioutput="raw_values")
        self.val_metric = R2Score(num_outputs=output_dim, multioutput="raw_values")
        self.test_metric = R2Score(num_outputs=output_dim, multioutput="raw_values")
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def step(self, batch, metric, step_type):
        x, y = batch
        predictions = self(x)
        mask = ~y.isnan()
        y_masked = y[mask].unsqueeze(1)
        predictions_masked = predictions[mask]
        loss = self.criterion(predictions_masked, y_masked)
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        if step_type == "val":
            r2 = metric(predictions_masked, y_masked).nanmean()
            self.log(f"{step_type}_metric", r2, prog_bar=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, self.train_metric, "train")

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, self.val_metric, "val")

    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, self.test_metric, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self(x)
        return predictions, y

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1)
        scheduler_config = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


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
