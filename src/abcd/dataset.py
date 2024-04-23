from torch import tensor
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def make_tensor_dataset(dataset, targets: str | list[str]):
    data = []
    for _, subject in dataset.groupby("src_subject_id"):
        match targets:
            case str():
                labels = subject.pop(targets)
            case list():
                labels = subject[targets]
                subject = subject.drop(labels, axis=1)
        subject.pop("src_subject_id")
        features = tensor(subject.to_numpy()).float()
        label = tensor(labels.to_numpy()).float()
        data.append((features, label))
    return data


class RNNDataset(Dataset):
    def __init__(self, dataset, targets: str) -> None:
        self.dataset = make_tensor_dataset(dataset, targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, labels = self.dataset[index]
        return features, labels


def collate(batch):
    sequence, label = zip(*batch)
    sequence = pad_sequence(sequence, batch_first=True)
    label = pad_sequence(label, batch_first=True, padding_value=float("nan"))
    return sequence, label.squeeze(2)


class ABCDDataModule(LightningDataModule):
    def __init__(
        self, train, val, test, batch_size, dataset_class, num_workers, targets
    ) -> None:
        super().__init__()
        self.train_dataset = dataset_class(dataset=train, targets=targets)
        self.val_dataset = dataset_class(dataset=val, targets=targets)
        self.test_dataset = dataset_class(dataset=test, targets=targets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate if dataset_class == RNNDataset else None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
