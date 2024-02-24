from torch import nn, tensor, concat, nan
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule, LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pack_sequence,
)
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from torchmetrics import R2Score
from abcdclinical.config import Config


def make_tensor_dataset(dataset):
    data = []
    for _, subject in dataset.groupby("src_subject_id"):
        p_score = subject.pop("p_score")
        subject.pop("src_subject_id")
        features = tensor(subject.to_numpy()).float()
        label = tensor(p_score.to_numpy()).float()
        data.append((features, label))
    return data


class RNNDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = make_tensor_dataset(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, labels = self.dataset[index]
        return features, labels


def collate(batch):
    sequences, labels = zip(*batch)
    packed_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=nan)
    return packed_sequences, labels


class ABCDDataModule(LightningDataModule):
    def __init__(self, train, val, test, batch_size, num_workers) -> None:
        super().__init__()
        self.train_dataset = RNNDataset(dataset=train)
        self.val_dataset = RNNDataset(dataset=val)
        self.test_dataset = RNNDataset(dataset=test)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True,
            collate_fn=collate,
        )


class Network(LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        weight_decay,
        lr,
        momentum,
        nesterov,
        batch_size,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)  # output_dim
        self.criterion = nn.MSELoss()
        self.optimizer = SGD(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min")
        self.r2_score = R2Score(num_outputs=4)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

    def step(self, batch):
        x, y = batch
        predictions = self(x)
        mask = ~y.isnan()
        y_masked = y[mask]
        predictions_masked = predictions[mask].squeeze(-1)
        loss = self.criterion(predictions_masked, y_masked)
        self.r2_score(predictions_masked, y_masked)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.step(train_batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)

    def validation_step(self, val_batch, batch_idx):
        loss = self.step(val_batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("r2", self.r2_score, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x), y

    def configure_optimizers(self):
        scheduler_config = {
            "scheduler": self.scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
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
    # early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    callbacks = [
        checkpoint_callback,
        # early_stopping,
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
