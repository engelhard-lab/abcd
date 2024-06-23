from torch import tensor
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import polars as pl


def make_tensor_dataset(dataset: pl.DataFrame):
    data = []
    for subject in dataset.partition_by(
        "src_subject_id", maintain_order=True, include_key=False
    ):
        label = subject.drop_in_place("label")
        label = tensor(label.to_numpy()).float()
        features = tensor(subject.to_numpy()).float()
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


class ABCDDataModule(LightningDataModule):
    def __init__(
        self, train, val, test, batch_size, dataset_class, num_workers
    ) -> None:
        super().__init__()
        self.train_dataset = dataset_class(dataset=train)
        self.val_dataset = dataset_class(dataset=val)
        self.test_dataset = dataset_class(dataset=test)
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        )
