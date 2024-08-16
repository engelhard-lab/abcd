from multiprocessing import cpu_count
from torch import tensor, long
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import polars as pl

from abcd.config import Config


def make_tensor_dataset(dataset: pl.DataFrame):
    data = []
    for subject in dataset.partition_by(
        "src_subject_id", maintain_order=True, include_key=False
    ):
        label = subject.drop_in_place("y_{t+1}")
        label = tensor(label.to_numpy()).float()
        features = tensor(subject.to_numpy()).float()
        data.append((features, label))
    return data


class RNNDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.dataset = make_tensor_dataset(df)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, labels = self.dataset[index]
        return features, labels


class EmbeddingDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.dataset = df.partition_by("src_subject_id")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        indices = tensor(data["index"].to_list(), dtype=long)
        labels = tensor(data["y_{t+1}"].to_numpy()).float()
        return indices, labels


def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=float("nan"))
    return padded_features, padded_labels


class ABCDDataModule(LightningDataModule):
    def __init__(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame,
        test: pl.DataFrame,
        batch_size: int,
        dataset_class,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dataset = dataset_class(df=train)
        self.val_dataset = dataset_class(df=val)
        self.test_dataset = dataset_class(df=test)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_names = train.drop(["src_subject_id", "y_t", "y_{t+1}"]).columns

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            collate_fn=collate_fn,
        )


def make_data_module(method: str, splits: dict, config: Config):
    match method:
        case "embedding":
            dataset_class = EmbeddingDataset
        case "rnn" | "mlp":
            dataset_class = RNNDataset
        case _:
            raise ValueError(
                f"Invalid method '{method}'. Choose from: 'rnn', 'mlp', or 'embedding'"
            )
    return ABCDDataModule(
        **splits,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=dataset_class,
    )
