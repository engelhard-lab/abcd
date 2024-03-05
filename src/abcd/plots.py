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
from torch import load as torch_load
from torch.utils.data import DataLoader
from shap import summary_plot
from matplotlib.patches import Patch

from abcd.dataset import RNNDataset, collate
from abcd.evaluate import RACE_MAPPING, SEX_MAPPING
from abcd.preprocess import DATASET_MAPPING


FORMAT = "png"

DATASET_NAMES = [
    "family_environment",
    "neighborhood",
    "problem_monitor",
    "prosocial",
    "school",
    "screentime",
    "sleep_disturbance",
    "rules",
    "brain_dti_fa",
    "brain_rsfmri",
    "brain_sst",
]
COLOR_MAPPING = dict(zip(DATASET_NAMES, sns.color_palette("tab20")))


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
    sns.catplot(
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


def predicted_vs_observed(hue):
    plt.figure()
    sns.set_palette(palette=sns.color_palette("tab20"))
    df = (
        pl.read_csv("data/results/predicted_vs_observed.csv")
        .with_columns(
            (pl.col("age") / 12).round(0).cast(pl.Int32),
            pl.col("race_ethnicity").replace(RACE_MAPPING),
            pl.col("sex").replace(SEX_MAPPING),
        )
        .rename(
            {
                "eventname": "Year",
                "race_ethnicity": "Race/Ethnicity",
                "sex": "Sex",
                "age": "Age",
            }
        )
    )
    df = (
        df.melt(
            id_vars=["y_pred", "y_true"],
            value_vars=["Year", "Race/Ethnicity", "Sex", "Age"],
        )
        .rename({"value": "Group", "variable": "Variable"})
        .sort("Group")
    )
    min_val = df.select(pl.min_horizontal(["y_true", "y_pred"]).min()).item()
    max_val = df.select(pl.max_horizontal(["y_true", "y_pred"]).max()).item()
    df = df.to_pandas()
    g = sns.lmplot(
        data=df,
        x="y_true",
        y="y_pred",
        hue="Group",
        col="Variable",
        col_wrap=2,
        facet_kws={"legend_out": False},
    )
    g.set(xlabel="Observed", ylabel="Predicted")
    for ax in g.axes.flat:
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        # handles = [
        #     Patch(facecolor=color) for color in sns.color_palette("deep")[: len(labels)]
        # ]
        ax.legend(
            handles=handles,
            labels=labels,
            title=ax.get_title().replace("Variable = ", ""),
            loc="upper left",
        )
        ax.set_title("")
        # r_squared = r2_score(group["y_true"], group["y_pred"])
    # sns.move_legend(g, title=hue.replace("_", "/").capitalize(), loc="center right")
    # plt.text(
    #     x=min_val,
    #     y=max_val,
    #     s=f"Mean R$^2$ = {r_squared:.2f}",
    #     fontsize=12,
    #     verticalalignment="top",
    # )
    # plt.axis("equal")
    plt.show()
    # plt.tight_layout()
    # plt.savefig(f"data/plots/predicted_vs_observed_{hue}.png", format=FORMAT)


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


def shap_plot(shap_values, X, feature_names):
    pass


def shap_by_year_plot(shap_values_list, X, feature_names):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    for i, (shap_values, ax) in enumerate(zip(shap_values_list, axes.flatten())):
        shap_values = shap_values.reshape(-1, shap_values.shape[2])
        plt.sca(ax)
        ax.set_title(f"Year {i + 1}")
        summary_plot(
            shap_values,
            features=X,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
    plt.tight_layout()
    plt.show()


def make_column_mapping(join_on):
    columns = {
        DATASET_MAPPING[dataset]: pl.read_csv(
            "data/features/" + dataset + ".csv",
            null_values="",
            infer_schema_length=50_000,
        )
        .drop(join_on)
        .columns
        for dataset in DATASET_MAPPING
    }
    return {
        value: key.replace("_youth", "").replace("_parent", "")
        for key, values in columns.items()
        for value in values
    }


def grouped_shap_plot(shap_values, feature_names, column_mapping):
    df = pl.DataFrame(shap_values, schema=feature_names).transpose(
        include_header=True, header_name="variable"
    )
    df = (
        df.filter(pl.col("variable") != "eventname")
        .with_columns(pl.col("variable").replace(column_mapping).alias("dataset"))
        .group_by("dataset")
        .sum()
        .drop("variable")
    )
    columns = df.drop_in_place("dataset")
    df = df.transpose(column_names=columns).to_numpy()
    summary_plot(df, feature_names=columns.to_list(), show=True)


def shap_clustermap(shap_values, feature_names, column_mapping):
    column_colors = {
        col: COLOR_MAPPING[dataset] for col, dataset in column_mapping.items()
    }
    colors = [column_colors[col] for col in feature_names[1:]]
    shap_df = pl.DataFrame(shap_values[:, 1:], schema=feature_names[1:]).to_pandas()
    shap_corr = shap_df.corr()
    g = sns.clustermap(
        shap_corr,
        row_colors=colors,
        yticklabels=False,
        xticklabels=False,
    )
    g.ax_col_dendrogram.set_visible(False)
    mask = np.triu(np.ones_like(shap_corr))
    values = g.ax_heatmap.collections[0].get_array().reshape(shap_corr.shape)  # type: ignore
    new_values = np.ma.array(values, mask=mask)
    g.ax_heatmap.collections[0].set_array(new_values)
    handles = [Patch(facecolor=color) for color in COLOR_MAPPING.values()]
    plt.legend(
        handles,
        COLOR_MAPPING.keys(),
        title="Dataset",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    plt.show()


def plot(dataloader):
    sns.set_theme(font_scale=1.5, style="whitegrid", palette="deep")
    # feature_names = (
    #     pl.read_csv("data/analytic/test.csv")
    #     .drop(["src_subject_id", "p_score"])
    #     .columns
    # )
    # test_dataloader = iter(dataloader)
    # X, _ = next(test_dataloader)
    # X = X.view(-1, X.shape[2])
    # shap_values_list = torch_load("data/results/shap_values.pt")
    # shap_values = np.mean(shap_values_list, axis=0)
    # shap_values = shap_values.reshape(-1, shap_values.shape[2])
    # assert X.shape[1] == len(feature_names)
    # summary_plot(
    #     shap_values,
    #     features=X,
    #     show=True,
    # )
    # shap_by_year_plot(
    #     shap_values_list=shap_values_list, X=X, feature_names=feature_names
    # )
    # column_mapping = make_column_mapping(join_on=["src_subject_id", "eventname"])
    # grouped_shap_plot(
    #     shap_values=shap_values,
    #     feature_names=feature_names,
    #     column_mapping=column_mapping,
    # )
    # shap_clustermap(
    #     shap_values=shap_values,
    #     feature_names=feature_names,
    #     column_mapping=column_mapping,
    # )

    # make_coefficent_table(model)
    # coefficients = pl.read_csv("data/results/coefficients.csv")
    predicted_vs_observed(hue="race_ethnicity")
    # predicted_vs_observed(hue="sex")
    # predicted_vs_observed(hue="eventname")
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
    test = pd.read_csv("data/analytic/test.csv")
    test_dataset = RNNDataset(dataset=test, target="p_score")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=500,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    plot(dataloader=test_dataloader)
