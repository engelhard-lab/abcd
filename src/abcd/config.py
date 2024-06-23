from pydantic import BaseModel
from pathlib import Path


class Filepaths(BaseModel):
    data: Path
    features: Path
    labels: Path
    checkpoints: Path
    study: Path
    logs: Path
    analytic: Path
    train: Path
    val: Path
    test: Path
    predictions: Path

    def __init__(self, **data):
        super().__init__(**data)
        self.features = self.data / self.features
        self.labels = self.data / self.labels
        self.checkpoints = self.data / self.checkpoints
        self.study = self.data / self.study
        self.logs = self.data / self.logs
        self.analytic = self.data / self.analytic
        self.train = self.analytic / self.train
        self.val = self.analytic / self.val
        self.test = self.analytic / self.test
        self.predictions = self.data / self.predictions


class Preprocess(BaseModel):
    n_neighbors: int
    n_quantiles: int
    train_size: float


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
    random_seed: int
    regenerate: bool
    predict: bool
    tune: bool
    evaluate: bool
    shap: bool
    plot: bool
    tables: bool
    n_trials: int
    verbose: bool
    n_trials: int
    join_on: list[str]
    target: str
    filepaths: Filepaths
    preprocess: Preprocess
    labels: Labels
    features: Features
    training: Training
    logging: Logging
    optimizer: Optimizer
    model: Model

    def __init__(self, **data):
        super().__init__(**data)
