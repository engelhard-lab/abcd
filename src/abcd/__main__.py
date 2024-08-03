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
from abcd.preprocess import get_raw_dataset, process_splits
from abcd.config import Config, update_paths
from abcd.model import make_model, make_trainer
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune
from abcd.utils import cleanup_checkpoints

analyses = ["symptoms", "with_brain", "by_year", "without_brain", "all"]


def main():
    pl.Config(tbl_cols=14)
    set_config(transform_output="polars")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    splits = get_raw_dataset(config=config)
    for analysis in analyses:
        print(analysis)
        analysis_path = "data/analyses" / Path(analysis)
        with open("config.toml", "rb") as f:
            config = Config(**load(f))
        update_paths(config, new_path=analysis_path)
        train, val, test = process_splits(
            splits=splits, config=config, analysis=analysis
        )
        data_module = ABCDDataModule(
            train=train,
            val=val,
            test=test,
            batch_size=config.training.batch_size,
            num_workers=cpu_count(),
            dataset_class=RNNDataset,
        )
        input_dim = next(iter(data_module.train_dataloader()))[0].shape[-1]
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
        print(config.filepaths.data.results.checkpoints)
        print(config.filepaths.data.results.study)
        model = make_model(
            input_dim=input_dim,
            output_dim=config.preprocess.n_quantiles,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            checkpoints=None,  # , config.filepaths.data.results.checkpoints,
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
    # if config.shap:
    #     make_shap(train, model=model, data_module=data_module)
    # if config.tables:
    #     make_tables()
    # if config.plot:
    #     plot(config=config)


if __name__ == "__main__":
    main()
