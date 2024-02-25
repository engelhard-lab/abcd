from tomllib import load
import pickle
from sklearn import set_config

from abcdclinical.dataset import ABCDDataModule, RNNDataset
from abcdclinical.evaluate import evaluate

from abcdclinical.preprocess import get_data
from abcdclinical.config import Config
from abcdclinical.model import Network, make_trainer

# from abcdclinical.plots import (
#     coefficients_plot,
#     elbow_plot,
#     pc_loadings_plot,
#     pc_mean_plot,
#     plot_random_intercept,
#     plot_residuals,
#     predicted_vs_observed,
# )
from abcdclinical.tune import tune
from abcdclinical.utils import cleanup_checkpoints, get_best_checkpoint


def main():
    set_config(transform_output="pandas")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    train, val, test = get_data(config, regenerate=config.regenerate)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=0,
        dataset_class=RNNDataset,
        target="p_score",
    )
    input_dim = train.shape[1] - 2
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
        )
        best_model_path = get_best_checkpoint(
            ckpt_folder=config.filepaths.checkpoints, mode="min"
        )
        model = Network.load_from_checkpoint(best_model_path)
    else:
        with open("data/studies/study.pkl", "rb") as f:
            study = pickle.load(f)
        if config.refit:
            model = Network(
                input_dim=input_dim,
                output_dim=1,  # FIXME move to config
                momentum=config.optimizer.momentum,
                nesterov=config.optimizer.nesterov,
                # batch_size=config.training.batch_size,
                **study.best_params,
            )
            trainer, _ = make_trainer(config)
            trainer.fit(model, datamodule=data_module)
            cleanup_checkpoints(config.filepaths.checkpoints, mode="min")
        else:
            best_model_path = get_best_checkpoint(
                ckpt_folder=config.filepaths.checkpoints, mode="min"
            )
            model = Network.load_from_checkpoint(best_model_path)
    print(study.best_params)

    if config.evaluate:
        evaluate(config=config, model=model, data_module=data_module)

    # sns.set_theme(font_scale=1.5, style="whitegrid")

    # make_tables()

    # make_coefficent_table(model)
    # coefficients = pl.read_csv("data/results/coefficients.csv")
    # predicted_vs_observed(y_test, y_pred)
    # plot_residuals(y_test, y_pred)
    # coefficients_plot()
    # plot_random_intercept(model)

    # indices = coefficients["index"].drop_nulls().drop_nans()[:20].cast(pl.Utf8)
    # components = pl.read_csv("data/results/principal_components.csv")
    # largest_components = components.select("name", "dataset", pl.col(indices))
    # pc_mean_plot(largest_components)
    # pc_loadings_plot(largest_components)
    # elbow_plot()


if __name__ == "__main__":
    main()
