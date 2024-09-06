from functools import partial
from lightning import LightningDataModule
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import polars as pl
import torch
from tqdm import tqdm
from captum.attr import GradientShap

from abcd.config import Config, get_config


def predict(x, model):
    output = model(x)
    return output.sum(dim=1)[:, -1]


def make_shap_values(model, X, background, columns):
    f = partial(predict, model=model.to("mps:0"))
    explainer = GradientShap(f)
    shap_values = explainer.attribute(X.to("mps:0"), background.to("mps:0"))
    return pl.DataFrame(shap_values.flatten(0, 1).cpu().numpy(), schema=columns)


def regress_shap_values(shap_values, X, n_bootstraps: int):
    coefs = {name: [] for name in X.columns}
    for _ in tqdm(range(n_bootstraps)):
        X_resampled, shap_resampled = resample(X, shap_values)  # type: ignore
        for col in shap_values.columns:
            coef = (
                make_pipeline(StandardScaler(), LinearRegression())
                .fit(X_resampled[[col]], shap_resampled[[col]])
                .named_steps["linearregression"]
                .coef_[0, 0]
            )
            coefs[col].append(coef)
    df = pl.DataFrame(coefs).melt()
    return df


def make_shap(
    config: Config,
    model,
    data_module: LightningDataModule,
    analysis: str,
    factor_model: str,
):
    config = get_config(analysis=analysis, factor_model=factor_model)
    test_dataloader = iter(data_module.test_dataloader())
    X = torch.cat([x for x, _ in test_dataloader])
    val_dataloader = iter(data_module.val_dataloader())
    background = torch.cat([x for x, _ in val_dataloader])
    feature_names = (
        pl.read_csv(config.filepaths.data.analytic.test)
        .drop(["src_subject_id", "y_{t+1}"])
        .columns
    )
    shap_values = make_shap_values(model, X, background, columns=feature_names)
    features = pl.DataFrame(X.flatten(0, 1).numpy(), schema=feature_names)
    df = regress_shap_values(shap_values, X=features, n_bootstraps=config.n_bootstraps)
    path = f"data/analyses/{factor_model}/{analysis}/results/"
    shap_values.write_csv(path + "shap_values.csv")
    features.write_csv(path + "shap_features.csv")
    df.write_csv(path + "shap_coefficients.csv")
