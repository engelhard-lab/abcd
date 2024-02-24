import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


FORMAT = "png"


def elbow_plot():
    plt.figure()
    X = pd.read_csv("data/analytic/raw_train_features.csv")
    pipeline_steps = [
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="median")),
        ("pca", PCA()),
    ]
    pipeline = Pipeline(steps=pipeline_steps)
    X = pipeline.fit_transform(X)
    component_names = range(1, X.shape[1] + 1)
    data = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pipeline.named_steps[
                "pca"
            ].explained_variance_ratio_,
        }
    )
    g = sns.lineplot(data=data, x="component", y="explained_variance_ratio", alpha=0.5)
    g.set(ylabel="Explained variance ratio", xlabel="Principal component")
    plt.tight_layout()
    plt.savefig("data/plots/elbow.png", format=FORMAT)


def pc_mean_plot(components):
    plt.figure()
    data = (
        pl.DataFrame(components)
        .melt(id_vars=["name", "dataset"])
        .with_columns(
            pl.col("dataset").str.replace("_youth", "").str.replace("_parent", "")
        )
        .with_columns(pl.col("value").abs())
        .drop_nulls()
        # .with_columns(pl.col("variable").cast(pl.Int32))
        .rename({"variable": "PC"})
    )
    g = sns.catplot(
        data=data,
        x="value",
        y="dataset",
        col="PC",
        col_wrap=4,
        kind="bar",
        sharex=False,
    )
    g.set_axis_labels("Mean component loading", "Dataset")
    plt.savefig("data/plots/pc_loading_mean.png", format=FORMAT)


def pc_loadings_plot(components):
    plt.figure()
    components = pl.DataFrame(components)
    data = (
        components.melt(id_vars=["name", "dataset"])
        .group_by("variable", maintain_order=True)
        .map_groups(lambda x: x.sort(pl.col("value").abs(), descending=True).head(10))
    )
    g = sns.catplot(
        x="value",
        y="name",
        hue="dataset",
        col="variable",
        col_wrap=4,
        kind="bar",
        data=data,
        sharey=False,
        sharex=False,
    )
    plt.savefig("data/plots/pc_loading.png", format=FORMAT)


def predicted_vs_observed(y_test, y_pred):
    plt.figure()
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    r_squared = r2_score(y_test, y_pred)
    g = sns.scatterplot(x=y_test, y=y_pred, color="#4C72B0", marker="o")
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    g.set(xlabel="Observed", ylabel="Predicted")
    plt.text(
        x=min_val,
        y=max_val,
        s=f"R$^2$ = {r_squared:.2f}",
        fontsize=12,
        verticalalignment="top",
    )
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("data/plots/predicted_vs_observed.png", format=FORMAT)


def plot_residuals(y_test, y_pred):
    plt.figure()
    g = sns.histplot(x=np.abs(y_test - y_pred), color="#4C72B0")
    g.set(xlabel="Residuals", ylabel="Frequency")
    plt.savefig("data/plots/residuals.png", format=FORMAT)


def coefficients_plot():
    plt.figure(figsize=(8, 10), dpi=300)
    df = pd.read_csv("data/results/coefficients.csv").head(20)
    g = sns.scatterplot(x="Coef.", y="Predictor", data=df, color="black", zorder=2)
    g.set(
        title="PCA Regression Coefficients",
        xlabel="Coefficient value",
        ylabel="Principal components",
    )
    for i in range(df.shape[0]):
        plt.plot(
            [df["[0.025"][i], df["0.975]"][i]],
            [df["Predictor"][i], df["Predictor"][i]],
            color="grey",
            zorder=1,
        )
    plt.axvline(x=0.0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig("data/plots/coefficients.png", format=FORMAT)


def plot_random_intercept(model):
    plt.figure()
    df = pl.DataFrame(model.random_effects).melt().to_pandas()
    g = sns.histplot(x=df["value"], color="#4C72B0")
    g.set(xlabel="Random intercept")
    plt.savefig("data/plots/random_intercept.png", format=FORMAT)
