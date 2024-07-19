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
from abcd.preprocess import get_data
from abcd.config import Config, update_paths
from abcd.model import make_model
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune

analyses = ["with_brain", "questionaires", "cbcl", "all", "by_year"]


def main():
    pl.Config(tbl_cols=14)
    set_config(transform_output="polars")

    for analysis in analyses:
        pass

    with open("config.toml", "rb") as f:
        config = Config(**load(f))

    update_paths(config, Path("example"))
    seed_everything(config.random_seed)

    train, val, test = get_data(config=config, analysis=analysis)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
    )
    input_dim = train.shape[1] - 2  # -2 for src_subject_id and label
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
            output_dim=config.preprocess.n_quantiles,
        )
    else:
        with open(config.filepaths.results.study, "rb") as f:
            study: Study = pickle.load(f)
    print(study.best_params)
    model = make_model(
        input_dim=input_dim,
        output_dim=config.preprocess.n_quantiles,
        momentum=config.optimizer.momentum,
        nesterov=config.optimizer.nesterov,
        checkpoints=config.filepaths.results.checkpoints,
        **study.best_params,
    )
    if config.evaluate:
        evaluate_model(config=config, model=model)
    if config.shap:
        make_shap(train, model=model, data_module=data_module)
    if config.tables:
        make_tables()
    if config.plot:
        plot(config=config)  # , dataloader=data_module.test_dataloader()


if __name__ == "__main__":
    main()
