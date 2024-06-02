from pydantic import BaseModel
from pathlib import Path


class Filepaths(BaseModel):
    data: Path
    features: Path
    labels: Path
    checkpoints: Path
    study: Path
    logs: Path
    train: Path
    val: Path
    test: Path

    def __init__(self, **data):
        super().__init__(**data)
        self.features = self.data / self.features
        self.labels = self.data / self.labels
        self.checkpoints = self.data / self.checkpoints
        self.study = self.data / self.study
        self.logs = self.data / self.logs
        self.train = self.data / self.train
        self.val = self.data / self.val
        self.test = self.data / self.test


class Training(BaseModel):
    batch_size: int
    max_epochs: int
    gradient_clip: float


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


class Data(BaseModel):
    name: str
    respondent: str
    columns: list[str]


class Labels(BaseModel):
    p_factor: str
    cbcl_labels: list[str]


class Features(BaseModel):
    abcd_p_demo: Data
    led_l_adi: Data
    abcd_y_lt: Data
    abcd_aces: Data
    ce_p_fes: Data
    ce_y_fes: Data
    ce_p_nsc: Data
    ce_y_nsc: Data
    ce_y_pm: Data
    ce_p_psb: Data
    ce_y_psb: Data
    ce_y_srpf: Data
    nt_p_stq: Data
    nt_y_st: Data
    ph_p_sds: Data
    su_p_pr: Data
    mri_y_dti_fa_fs_at: Data
    mri_y_rsfmr_cor_gp_gp: Data
    mri_y_tfmr_sst_csvcg_dsk: Data
    mri_y_tfmr_mid_alrvn_dsk: Data
    mri_y_tfmr_nback_2b_dsk: Data


class Config(BaseModel):
    fast_dev_run: bool
    name: str
    random_seed: int
    regenerate: bool
    tune: bool
    refit: bool
    evaluate: bool
    plot: bool
    tables: bool
    n_trials: int
    verbose: bool
    train_size: float
    n_trials: int
    method: str
    join_on: list[str]
    target: str
    n_quantiles: int
    filepaths: Filepaths
    labels: Labels
    features: Features
    training: Training
    logging: Logging
    optimizer: Optimizer
    model: Model

    def __init__(self, **data):
        super().__init__(**data)
        self.filepaths.study = self.filepaths.study / (self.name + ".pkl")
