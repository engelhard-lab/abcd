from multiprocessing import cpu_count

from lightning import seed_everything
from sklearn import set_config
import polars as pl
from tqdm import tqdm
from itertools import product

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap
from abcd.preprocess import get_dataset
from abcd.config import get_config
from abcd.model import make_model
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune


def main():
    pl.Config(tbl_cols=14)
    set_config(transform_output="polars")
    config = get_config()
    analyses = product(config.analyses, config.factor_models)
    progress_bar = tqdm(
        analyses, total=len(config.analyses) * len(config.factor_models)
    )
    for analysis, factor_model in progress_bar:
        seed_everything(config.random_seed)
        config = get_config(analysis=analysis, factor_model=factor_model)
        splits = get_dataset(
            analysis=analysis, factor_model=factor_model, config=config
        )
        data_module = ABCDDataModule(
            **splits,
            batch_size=config.training.batch_size,
            num_workers=cpu_count(),
            dataset_class=RNNDataset,
        )
        input_dim = 1 if analysis == "autoregressive" else splits["train"].shape[-1] - 2
        if config.tune:
            tune(
                config=config,
                data_module=data_module,
                input_dim=input_dim,
                output_dim=config.preprocess.n_quantiles,
            )
        model = make_model(
            input_dim=input_dim,
            output_dim=config.preprocess.n_quantiles,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            checkpoints=config.filepaths.data.results.checkpoints,
        )
        if config.evaluate:
            evaluate_model(data_module=data_module, config=config, model=model)
    if config.shap:
        make_shap(
            config=config, model=model, data_module=data_module, analysis=analysis
        )
    if config.tables:
        make_tables(config=config)
    if config.plot:
        plot(config=config)


if __name__ == "__main__":
    main()
