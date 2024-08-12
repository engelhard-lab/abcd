from multiprocessing import cpu_count
from pathlib import Path
from tomllib import load
import pickle

from lightning import seed_everything
from optuna import Study
from sklearn import set_config
import polars as pl

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap
from abcd.metadata import make_subject_metadata
from abcd.preprocess import (
    generate_data,
    get_dataset,
    get_features,
    split_data,
)
from abcd.config import Config, update_paths
from abcd.model import make_model, make_trainer
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune
from abcd.utils import cleanup_checkpoints

# TODO , "by_year", "metadata",
analyses = ["all", "with_brain", "symptoms", "without_brain"]


def main():
    pl.Config(tbl_cols=14)
    set_config(transform_output="polars")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    if config.regenerate:
        generate_data(config=config)
    df = pl.read_csv(config.filepaths.data.raw.dataset)
    for analysis in analyses:
        print(analysis)
        with open("config.toml", "rb") as f:
            config = Config(**load(f))
        analysis_path = "data/analyses" / Path(analysis)
        update_paths(config, new_path=analysis_path)
        feature_selection = get_features(analysis=analysis, config=config)
        feature_columns = df.select(feature_selection).columns
        print(feature_columns)
        splits = split_data(
            df=df,
            group=config.join_on[0],
            train_size=config.preprocess.train_size,
            random_state=config.random_seed,
        )
        splits = get_dataset(splits, feature_columns=feature_columns, config=config)
        if analysis == "all":
            metadata = make_subject_metadata(splits=splits)
            metadata.write_csv(config.filepaths.data.raw.metadata)
        data_module = ABCDDataModule(
            **splits,
            batch_size=config.training.batch_size,
            num_workers=cpu_count(),
            dataset_class=RNNDataset,
        )
        input_dim = splits["train"].shape[-1] - 3
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
        model = make_model(
            input_dim=input_dim,
            output_dim=config.preprocess.n_quantiles,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            checkpoints=config.filepaths.data.results.checkpoints,
            **study.best_params,
        )
        if config.refit_best:
            trainer, _ = make_trainer(config=config)
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
