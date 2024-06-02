from multiprocessing import cpu_count
from tomllib import load
import pickle
from lightning import seed_everything
from optuna import Study
import shap
from sklearn import set_config
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
from sklearn.utils import resample
import torch
from torchmetrics.functional import auroc
from tqdm import tqdm

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model, make_predictions
from abcd.importance import make_shap_values, regress_shap_values
from abcd.preprocess import get_data
from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune


def main():
    set_config(transform_output="polars")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    train, val, test = get_data(config)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
    )
    output_dim = config.n_quantiles
    input_dim = train.shape[1] - 3  # src_subject_id, next_p_factor, p_factor
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        with open(config.filepaths.study, "rb") as f:
            study: Study = pickle.load(f)
    print(study.best_params)
    model = make_model(
        input_dim=input_dim,
        output_dim=output_dim,
        momentum=config.optimizer.momentum,
        nesterov=config.optimizer.nesterov,
        checkpoints=config.filepaths.checkpoints,
        **study.best_params,
    )

    if config.evaluate:
        evaluate_model(config=config, model=model)

        # trainer, _ = make_trainer(config)
        # metrics = trainer.test(model, dataloaders=data_module.test_dataloader())
        # print(metrics)

        # auc_subsets(model, config)

        # predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
        # outputs, labels = zip(*predictions)
        # outputs = torch.concat(outputs)
        # labels = torch.concat(labels).long().squeeze(-1)
        # aucs = []
        # n_bootstraps = 1000
        # for _ in tqdm(range(n_bootstraps)):
        #     outputs_resampled, labels_resampled = resample(outputs, labels)
        #     auc = auroc(
        #         outputs_resampled,
        #         labels_resampled,
        #         task="multiclass",
        #         num_classes=outputs.shape[-1],
        #         average="none",
        #     )
        #     aucs.append({f"auc_{i+1}": value for i, value in enumerate(auc)})
        # df = pl.DataFrame(aucs)
        # df.write_csv("data/results/aucs.csv")

        # outputs = torch.concat(outputs).flatten(0, 1)
        # labels = torch.concat(labels).flatten(0, 2).long()
        # events = np.tile(np.arange(1, 5), reps=outputs.shape[0])
        # subjects = np.repeat(np.arange(outputs.shape[0]), repeats=4)
        # auc = auroc(
        #     outputs,
        #     labels,
        #     task="multiclass",
        #     num_classes=outputs.shape[-1],
        #     average="none",
        # )
        # print(auc)
        # labels = pl.Series(labels.numpy(), dtype=pl.Int64)
        # results = pl.DataFrame(outputs.numpy())
        # results = results.with_columns(labels=labels)
        # moves_into_4th_quartile = labels.eq(3) & ~labels.shift(1).eq(3)
        # moves_out_of_4th_quartile = labels.ne(3) & labels.shift(1).eq(3)
        # print(metrics)
        # auroc = pl.DataFrame(metrics).select(cs.contains("auroc"))
        # name_map = {name: name.replace("auroc_", "") for name in auroc.columns}
        # auroc = (
        #     auroc.rename(name_map)
        #     .melt()
        #     .with_columns(pl.col("variable").cast(pl.Int32).add(1))
        #     .rename({"value": "AUROC", "variable": "Quartile"})
        # )
        # print(auroc)
        # columns = list(train.drop(["src_subject_id", "p_factor"], axis=1).columns)
        # test_dataloader = iter(data_module.test_dataloader())
        # X, labels = next(test_dataloader)
        # background, _ = next(test_dataloader)
        # shap_values = make_shap_values(model, X, background, columns=columns)
        # shap_values.write_csv("data/results/shap_values.csv")
        # features = pl.DataFrame(X.flatten(0, 1).numpy(), schema=columns)
        # if False:
        #     labels = labels.flatten(0, 2).long().numpy()
        #     labels = pl.Series(labels, dtype=pl.Int64)
        #     moves_to_4th_quartile = labels.eq(3) & ~labels.shift(1).eq(3)
        #     shap_values = shap_values.filter(moves_to_4th_quartile)
        #     features = features.filter(moves_to_4th_quartile)
        # if True:
        #     sex = features["demo_sex_v2_1"] > 0
        #     male_shap_values = shap_values.filter(sex).with_columns(
        #         pl.lit("Male").alias("Sex")
        #     )
        #     female_shap_values = shap_values.filter(~sex).with_columns(
        #         pl.lit("Female").alias("Sex")
        #     )
        #     sex_coefs = pl.concat([male_shap_values, female_shap_values])
        #     sex_coefs.write_csv("data/results/sex_shap_coefs.csv")

        # df = regress_shap_values(shap_values, X=features)
        # df.write_csv("data/results/shap_coefs.csv")
        # evaluate_model(
        #     targets=targets, config=config, model=model, data_module=data_module
        # )
    if config.plot:
        plot(dataloader=data_module.test_dataloader())
    if config.tables:
        make_tables()


if __name__ == "__main__":
    main()
