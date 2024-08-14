from multiprocessing import cpu_count
from tomllib import load
import pickle

from lightning import seed_everything
from optuna import Study
from sklearn import set_config
import polars as pl
from tqdm import tqdm

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap
from abcd.preprocess import get_dataset
from abcd.config import Config, update_paths
from abcd.model import make_model, make_trainer
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune
from abcd.utils import cleanup_checkpoints

analyses = [
    # "by_year",
    # "questions",
    # "questions_brain",
    # "symptoms",
    # "questions_symptoms",
    # "all",
    "autoregressive",
]


def main():
    pl.Config(tbl_cols=14)
    set_config(transform_output="polars")
    for analysis in tqdm(analyses):
        with open("config.toml", "rb") as f:
            config = Config(**load(f))
        seed_everything(config.random_seed)
        update_paths(config, analysis=analysis)
        splits = get_dataset(analysis=analysis, config=config)
        data_module = ABCDDataModule(
            **splits,
            batch_size=config.training.batch_size,
            num_workers=cpu_count(),
            dataset_class=RNNDataset,
        )
        input_dim = 1 if analysis == "autoregressive" else splits["train"].shape[-1] - 2
        if config.tune:
            study = tune(
                config=config,
                data_module=data_module,
                input_dim=input_dim,
                output_dim=config.preprocess.n_quantiles,
            )
        else:
            with open(config.filepaths.data.results.study, "rb") as f:
                study: Study = pickle.load(f)
        print(study.best_params)
        checkpoint_path = (
            None if config.refit_best else config.filepaths.data.results.checkpoints
        )
        model = make_model(
            input_dim=input_dim,
            output_dim=config.preprocess.n_quantiles,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            checkpoints=checkpoint_path,
            **study.best_params,
        )
        if config.refit_best:
            trainer = make_trainer(config=config, checkpoint=True)
            trainer.fit(model=model, datamodule=data_module)
            cleanup_checkpoints(
                checkpoint_dir=config.filepaths.data.results.checkpoints, mode="min"
            )
        if config.evaluate:
            evaluate_model(data_module=data_module, config=config, model=model)
    if config.shap:
        make_shap(model=model, data_module=data_module)
    if config.tables:
        make_tables()
    if config.plot:
        plot(config=config)


if __name__ == "__main__":
    main()
