from pydantic import BaseModel
from pathlib import Path


class Filepaths(BaseModel):
    data: Path
    features: Path
    labels: Path
    checkpoints: Path
    logs: Path
    train: Path
    val: Path
    test: Path

    def __init__(self, **data):
        super().__init__(**data)
        self.features = self.data / self.features
        self.labels = self.data / self.labels
        self.checkpoints = self.data / self.checkpoints
        self.logs = self.data / self.logs
        self.train = self.data / self.train
        self.val = self.data / self.val
        self.test = self.data / self.test


class Training(BaseModel):
    batch_size: int
    max_epochs: int
    gradient_clip: float
    # swa_lr: float


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Optimizer(BaseModel):
    lr: dict[str, float | bool]
    weight_decay: dict[str, float | bool]
    momentum: float
    nesterov: bool


class Model(BaseModel):
    hidden_dim: list[int]
    num_layers: dict[str, int]
    dropout: dict[str, float]


class Config(BaseModel):
    fast_dev_run: bool
    random_seed: int
    tune: bool
    refit: bool
    evaluate: bool
    regenerate: bool
    n_trials: int
    verbose: bool
    train_size: float
    n_trials: int
    method: str
    filepaths: Filepaths
    training: Training
    logging: Logging
    optimizer: Optimizer
    model: Model
