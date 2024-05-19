from multiprocessing import cpu_count
from tomllib import load
import pickle
from lightning import seed_everything
from optuna import Study
import shap
from sklearn import set_config
import polars as pl
import pandas as pd
import numpy as np

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model
from abcd.importance import make_shap_values, regress_shap_values
from abcd.preprocess import get_data
from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune


def main():
    set_config(transform_output="pandas")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    train, val, test = get_data(config)
    # sex = test["demo_sex_v2_1"].unique()
    # test = test[test["demo_sex_v2_1"] == sex[1]]
    # print(test)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
        target=config.target,
    )
    output_dim = 1  # config.n_quantiles
    input_dim = train.shape[1] - 2  # -1 for src_subject_id -1 for target
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        with open("data/studies/study.pkl", "rb") as f:
            study: Study = pickle.load(f)
    model = make_model(
        # method=config.method,
        input_dim=input_dim,
        output_dim=output_dim,
        momentum=config.optimizer.momentum,
        nesterov=config.optimizer.nesterov,
        checkpoints=config.filepaths.checkpoints,
        **study.best_params,
    )
    print(study.best_params)
    if config.evaluate:
        # trainer, _ = make_trainer(config)
        # trainer.test(model, dataloaders=data_module.test_dataloader())
        columns = list(train.drop(["src_subject_id", "p_factor"], axis=1).columns)
        test_dataloader = iter(data_module.test_dataloader())
        X, _ = next(test_dataloader)
        background, _ = next(test_dataloader)
        shap_values = make_shap_values(model, X, background, columns=columns)
        shap_values.write_csv("data/results/shap_values.csv")
        df = pl.read_csv(
            "data/test_untransformed.csv",
            columns=["src_subject_id", "eventname", "p_factor", "demo_sex_v2_1"],
        ).rename({"eventname": "event"})
        subjects = (df.group_by("src_subject_id", maintain_order=True).first())[
            : X.shape[0]
        ]["src_subject_id"]
        subjects = pl.Series(np.repeat(subjects.to_numpy(), repeats=4))
        event = pl.Series(np.tile(np.arange(1, 5), reps=X.shape[0]))
        features = pl.DataFrame(X.mean(dim=1).numpy(), schema=columns)  # .sum(dim=1)
        features = features.with_columns(src_subject_id=subjects, event=event)
        shap_values = shap_values.with_columns(src_subject_id=subjects, event=event)
        shap_values = df.join(shap_values, on=["src_subject_id", "event"], how="inner")
        shap_values = shap_values.with_columns(
            pl.col("p_factor")
            .qcut(quantiles=4, labels=["1", "2", "3", "4"])
            .alias("quartile")
        )
        regress_shap_values(shap_values, X=features)
        # evaluate_model(
        #     targets=targets, config=config, model=model, data_module=data_module
        # )
    if config.plot:
        plot(config=config, dataloader=data_module.test_dataloader())
    if config.tables:
        make_tables()


if __name__ == "__main__":
    main()
