from pydantic import BaseModel
from pathlib import Path


class Data(BaseModel):
    features: Path
    labels: Path
    train: Path
    val: Path
    test: Path


class Results(BaseModel):
    checkpoints: Path
    study: Path
    logs: Path
    predictions: Path
    tables: Path
    plots: Path


class Filepaths(BaseModel):
    data: Data
    results: Results


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


class Dataset(BaseModel):
    name: str
    respondent: str
    columns: list[str]


class Labels(BaseModel):
    p_factor: str
    cbcl_labels: list[str]


class Features(BaseModel):
    abcd_p_demo: Dataset
    led_l_adi: Dataset
    abcd_y_lt: Dataset
    abcd_aces: Dataset
    ce_p_fes: Dataset
    ce_y_fes: Dataset
    ce_p_nsc: Dataset
    ce_y_nsc: Dataset
    ce_y_pm: Dataset
    ce_p_psb: Dataset
    ce_y_psb: Dataset
    ce_y_srpf: Dataset
    nt_p_stq: Dataset
    nt_y_st: Dataset
    ph_p_sds: Dataset
    su_p_pr: Dataset
    mh_p_fhx: Dataset
    mri_y_dti_fa_fs_at: Dataset
    mri_y_rsfmr_cor_gp_gp: Dataset
    mri_y_tfmr_sst_csvcg_dsk: Dataset
    mri_y_tfmr_mid_alrvn_dsk: Dataset
    mri_y_tfmr_nback_2b_dsk: Dataset


class Config(BaseModel):
    fast_dev_run: bool
    random_seed: int
    regenerate: bool
    predict: bool
    tune: bool
    log: bool
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


def update_paths(config: Config, new_path: Path):
    config.filepaths.data = Data(
        **{
            name: "data" / path
            for name, path in config.filepaths.data.model_dump().items()
        }
    )
    config.filepaths.results = Results(
        **{
            name: "results" / new_path / path
            for name, path in config.filepaths.results.model_dump().items()
        }
    )
    return config
