from tomllib import load
import pickle
from sklearn import set_config
import statsmodels.api as sm
import polars as pl

import seaborn as sns
from abcdclinical.analysis import make_coefficent_table

from abcdclinical.dataset import drop_demographics, get_data
from abcdclinical.config import Config
from abcdclinical.plots import (
    coefficients_plot,
    elbow_plot,
    pc_loadings_plot,
    pc_mean_plot,
    plot_random_intercept,
    plot_residuals,
    predicted_vs_observed,
)
from abcdclinical.tune import tune


def main():
    set_config(transform_output="pandas")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    if config.tune:
        study = tune(config)
        with open("data/studies/study.pkl", "wb") as f:
            pickle.dump(study, f)
    else:
        with open("data/studies/study.pkl", "rb") as f:
            study = pickle.load(f)
    print(study.best_params)
    (
        X_train,
        y_train,
        group_train,
        X_val,
        y_val,
        group_val,
        X_test,
        y_test,
        group_test,
    ) = get_data(
        config, regenerate=config.refit, n_components=study.best_params["n_components"]
    )
    X_train, X_val, X_test = drop_demographics(X_train, X_val, X_test)
    if config.refit:
        model = sm.MixedLM(y_train, X_train, groups=group_train).fit(method="lbfgs")
        with open("data/studies/model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        with open("data/studies/model.pkl", "rb") as f:
            model = pickle.load(f)
    print(model.summary())
    y_pred = model.predict(X_test)

    sns.set_theme(font_scale=1.5, style="whitegrid")

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
